import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style):
    if not image:
        raise gr.Error("Загрузите фото!")

    # ИСПОЛЬЗУЕМ МОДЕЛЬ ОТ stability-ai (SDXL) - ОНА ОФИЦИАЛЬНАЯ
    model_id = "stability-ai/sdxl:7762fdc030b82013f9613f791e03946777656729517172827725838048256335"

    try:
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, architectural photography",
                "num_inference_steps": 30,
                "guidance_scale": 7.5
            }
        )
        return output[0] if isinstance(output, list) else output
    except Exception as e:
        # Это выведет реальную причину, если что-то не так с токеном
        raise gr.Error(f"Ошибка API: {str(e)}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="Фото комнаты")
            style_drop = gr.Dropdown(choices=["Modern", "Scandinavian", "Luxury", "Industrial"], value="Modern", label="Стиль")
            run_btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            output_img = gr.Image(label="Результат")

    run_btn.click(fn=generate_design, inputs=[input_img, style_drop], outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
