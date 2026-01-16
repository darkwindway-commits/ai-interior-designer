import gradio as gr
import replicate
import os
import requests
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GUMROAD_TOKEN = os.getenv("GUMROAD_TOKEN")
PRODUCT_ID = os.getenv("GUMROAD_PRODUCT_ID")

# Временное хранилище IP-адресов (сбросится при перезагрузке сервера)
free_usage_tracker = {}

def verify_and_generate(image, style, license_key, request: gr.Request):
    client_ip = request.client.host # Получаем IP пользователя
    
    # --- ЛОГИКА БЕСПЛАТНЫХ ПОПЫТОК ---
    if not license_key:
        user_free_count = free_usage_tracker.get(client_ip, 0)
        
        if user_free_count >= 2:
            raise gr.Error("Ваши 2 бесплатные попытки закончились. Пожалуйста, введите лицензионный ключ для продолжения!")
        
        # Увеличиваем счетчик бесплатных попыток
        free_usage_tracker[client_ip] = user_free_count + 1
        print(f"IP {client_ip} использовал бесплатную попытку {user_free_count + 1}/2")
    
    # --- ЛОГИКА ПЛАТНОГО КЛЮЧА ---
    else:
        try:
            response = requests.post(
                "https://api.gumroad.com/v2/licenses/verify",
                data={
                    "product_id": PRODUCT_ID,
                    "license_key": license_key,
                    "increment_uses_count": "true"
                }
            )
            data = response.json()
            if not data.get("success"):
                raise gr.Error(f"Ошибка ключа: {data.get('message', 'Неверный код')}")
        except Exception as e:
            if "Ошибка" in str(e): raise e
            raise gr.Error(f"Проблема с проверкой оплаты: {str(e)}")

    # --- ОБЩАЯ ГЕНЕРАЦИЯ ---
    if not image:
        raise gr.Error("Загрузите фото комнаты!")

    try:
        model_id = "stability-ai/stable-diffusion-3.5-large"
        output = replicate.run(
            model_id,
            input={
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 8k",
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "cfg": 4.5
            }
        )
        return output[0]
    except Exception as e:
        raise gr.Error(f"Ошибка нейросети: {str(e)}")

# Интерфейс
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    gr.Markdown("**Акция:** 2 пробные генерации бесплатно! Для безлимита (50 шт) введите ключ.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Фото комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Luxury", "Minimalist"], 
                value="Modern", 
                label="2. Стиль"
            )
            key_input = gr.Textbox(
                label="3. Лицензионный ключ (оставьте пустым для пробы)", 
                placeholder="XXXX-XXXX-XXXX-XXXX"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН ✨", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    run_btn.click(
        fn=verify_and_generate, 
        inputs=[input_img, style_drop, key_input], 
        outputs=output_img
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
