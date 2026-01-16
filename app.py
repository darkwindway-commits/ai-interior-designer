import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем API токен (он уже должен быть в Railway Variables)
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Пожалуйста, загрузите фото вашей комнаты!")

    # ИСПОЛЬЗУЕМ ПРЯМОЙ ID МОДЕЛИ СО СКРИНШОТА image_048ce7.png
    model_id = "stability-ai/stable-diffusion-3.5-large"
    
    try:
        # Настройка входов согласно документации модели
        output = replicate.run(
            model_id,
            input={
                "prompt": f"A professional {style} interior design, high quality, photorealistic, architectural photography, 8k resolution",
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "cfg": 4.5  # Значение со скриншота image_04892c.png
            }
        )
        # Модель возвращает результат
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Это выведет реальную причину ошибки на экран
        raise gr.Error(f"Ошибка API Replicate: {str(e)}")

# Создаем интерфейс (совместим с вашей версией Gradio 3.50.2)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Фото комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Luxury", "Boho"], 
                value="Modern", 
                label="2. Выберите стиль"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Ваш результат")

    # Передаем ровно 2 аргумента: картинку и стиль
    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    # Запуск на порту 7860 для Railway
    demo.launch(server_name="0.0.0.0", server_port=7860)
