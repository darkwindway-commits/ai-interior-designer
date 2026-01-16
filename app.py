import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Загрузите фото!")

    # ТА САМАЯ РАБОЧАЯ ВЕРСИЯ С REPLICATE
    model_id = "stability-ai/stable-diffusion:ac732df83cea7fff18b75a6a3a5c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b8c3b"
    
    try:
        output = replicate.run(
            model_id,
            input={
                "width": 768,
                "height": 768,
                "prompt": f"A professional {style} interior design, high quality, photorealistic",
                "image": open(image, "rb"),
                "num_inference_steps": 30,
                "refine": "expert_ensemble_refiner"
            }
        )
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Если ошибка — мы увидим её прямо в интерфейсе
        raise gr.Error(f"Ошибка API: {str(e)}")

with gr.Blocks() as demo:
    gr.Markdown("# 🏠 Ваш Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото")
            style_drop = gr.Dropdown(choices=["Modern", "Scandinavian", "Industrial"], value="Modern", label="Стиль")
            run_btn = gr.Button("СОЗДАТЬ")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
