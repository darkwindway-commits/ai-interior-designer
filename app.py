import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Пожалуйста, загрузите фото!")

    # ОФИЦИАЛЬНАЯ РАБОЧАЯ МОДЕЛЬ SDXL
    model_id = "stability-ai/sdxl:7762fdc030b82013f9613f791e03946777656729517172827725838048256335"

    try:
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, highly detailed, photorealistic, 8k, architectural magazine style",
                "negative_prompt": "low quality, blurry, distorted room, bad furniture",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        # Возвращаем ссылку на картинку
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Теперь ошибка будет выводиться текстом, а не ломать программу
        raise gr.Error(f"Ошибка Replicate: {str(e)}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 Ваш AI Дизайнер")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото вашей комнаты")
            style_choice = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Boho", "Luxury"], 
                value="Modern", 
                label="Стиль"
            )
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    run_btn.click(fn=generate_design, inputs=[input_img, style_choice], outputs=output_img)

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
