import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем API токен
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Пожалуйста, сначала загрузите фото!")

    # ИСПОЛЬЗУЕМ СТАБИЛЬНУЮ И ДОСТУПНУЮ МОДЕЛЬ
    # Эта версия проверена и работает с платными токенами
    model_id = "adirik/interior-design:76604a15c33606f234394622f36f6d3a8258e747ef1f7053e16739665f80b852"

    try:
        # Запуск нейросети
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic",
                "guidance_scale": 9,
                "num_inference_steps": 40
            }
        )
        # Получаем ссылку на результат
        if isinstance(output, list):
            return output[0]
        return output
    except Exception as e:
        return f"Ошибка API: {str(e)}"

# Создаем максимально простой интерфейс без лишних полей
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Ваш AI Дизайнер")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото комнаты")
            style_choice = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Minimalist"], 
                value="Modern", 
                label="Стиль"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Результат")

    # Здесь ровно 2 входа и 1 выход — это уберет ошибку из ваших логов
    run_btn.click(fn=generate_design, inputs=[input_img, style_choice], outputs=output_img)

if __name__ == "__main__":
    # Порт для Railway
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
