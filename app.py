import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем API токен
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Пожалуйста, загрузите фото!")

    # ИСПОЛЬЗУЕМ ВЕРСИЮ, КОТОРУЮ ПОСОВЕТОВАЛ ГРОК (она стабильна)
    model_id = "stability-ai/stable-diffusion:ac732df83cea7fff18b75a6a3a5c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b"
    
    try:
        # Запуск нейросети
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, architectural photography",
                "negative_prompt": "low quality, blurry, distorted furniture",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        # Модель SDXL возвращает список ссылок, берем первую
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Выводим ошибку текстом, чтобы понимать, если что-то не так
        raise gr.Error(f"Ошибка API: {str(e)}")

# Простой интерфейс (совместим с вашей версией Gradio 3.50.2)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Luxury", "Industrial"], 
                value="Modern", 
                label="Стиль"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    # Здесь ровно 2 входа и 1 выход — это уберет прошлые ошибки в логах
    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
