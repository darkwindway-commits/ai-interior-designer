import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем API токен из переменных окружения
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        return None
    
    # ИСПОЛЬЗУЕМ ОФИЦИАЛЬНУЮ МОДЕЛЬ SDXL (она всегда доступна)
    model_id = "stability-ai/sdxl:7762fdc030b82013f9613f791e03946777656729517172827725838048256335"
    
    try:
        # Запускаем генерацию
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 4k",
                "negative_prompt": "low quality, blurry, distorted furniture",
                "num_inference_steps": 30
            }
        )
        # Получаем результат
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        raise gr.Error(f"Ошибка Replicate: {str(e)}")

# Интерфейс (совместимый с вашей версией Gradio 3.50.2)
with gr.Blocks() as demo:
    gr.Markdown("# 🏠 Ваш AI Дизайнер")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Luxury"], 
                value="Modern", 
                label="Стиль"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
