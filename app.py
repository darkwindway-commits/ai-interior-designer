import gradio as gr
import replicate
import os
import requests
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GUMROAD_TOKEN = os.getenv("GUMROAD_TOKEN")
PRODUCT_ID = os.getenv("GUMROAD_PRODUCT_ID")

def verify_and_generate(image, style, license_key):
    # 1. Базовые проверки входных данных
    if not license_key:
        raise gr.Error("Введите лицензионный ключ, полученный после оплаты!")
    if not image:
        raise gr.Error("Пожалуйста, загрузите фотографию комнаты.")

    # 2. Проверка ключа через Gumroad API
    try:
        # Отправляем запрос на верификацию и списываем 1 использование
        response = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": PRODUCT_ID,
                "license_key": license_key,
                "increment_uses_count": "true" 
            }
        )
        data = response.json()
        
        # Если API вернуло ошибку (ключ не существует или лимит исчерпан)
        if not data.get("success"):
            error_msg = data.get("message", "Неверный ключ")
            raise gr.Error(f"Ошибка доступа: {error_msg}")
            
        # Дополнительная проверка лимита (на всякий случай)
        uses = data.get("uses", 0)
        if uses > 50:
            raise gr.Error("Лимит этого ключа (50 генераций) полностью исчерпан.")
            
    except Exception as e:
        if "Ошибка доступа" in str(e): raise e
        raise gr.Error(f"Не удалось проверить оплату: {str(e)}")

    # 3. Генерация дизайна через Replicate
    try:
        # Используем модель Stable Diffusion 3.5 Large
        model_id = "stability-ai/stable-diffusion-3.5-large"
        
        # Формируем промпт на основе выбранного стиля
        prompt = f"A professional {style} interior design, high quality, photorealistic, 8k, architectural photography"
        
        output = replicate.run(
            model_id,
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "cfg": 4.5
            }
        )
        # Возвращаем первую сгенерированную картинку
        return output[0]
        
    except Exception as e:
        raise gr.Error(f"Ошибка нейросети: {str(e)}")

# Создание современного веб-интерфейса
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    gr.Markdown("Загрузите фото, выберите стиль и введите ваш ключ для мгновенного преображения.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Фотография комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Luxury", "Industrial", "Boho", "Minimalist"], 
                value="Modern", 
                label="2. Выберите стиль дизайна"
            )
            key_input = gr.Textbox(
                label="3. Лицензионный ключ", 
                placeholder="Вставьте ваш код из письма после оплаты",
                type="password" # Скрывает ключ при вводе
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
            
        with gr.Column():
            output_img = gr.Image(label="Ваш новый интерьер")

    # Логика работы кнопки
    run_btn.click(
        fn=verify_and_generate, 
        inputs=[input_img, style_drop, key_input], 
        outputs=output_img
    )

# Запуск приложения
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
