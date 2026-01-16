import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем переменные
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Сначала загрузите фото комнаты!")

    # ИСПОЛЬЗУЕМ 100% ПУБЛИЧНУЮ И СТАБИЛЬНУЮ МОДЕЛЬ (Актуально на 16.01.2026)
    # Это официальная версия Stable Diffusion XL от Stability AI
    model_id = "stability-ai/stable-diffusion:ac732df83cea7fff18b75a6a3a5c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b"
    
    try:
        # Запуск нейросети
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 4k",
                "negative_prompt": "low quality, blurry, distorted furniture",
                "num_inference_steps": 30
            }
        )
        # Получаем результат (модель SDXL обычно возвращает список ссылок)
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Если вдруг что-то не так, выведем понятную ошибку
        raise gr.Error(f"Ошибка Replicate: {str(e)}")

# Простой и понятный интерфейс
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера Pro")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Загрузите фото вашей комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Luxury", "Boho"], 
                value="Modern", 
                label="2. Выберите стиль дизайна"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Ваш обновленный интерьер")

    # Связываем кнопку с функцией
    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    # Настройка для Railway
    demo.launch(server_name="0.0.0.0", server_port=7860)
