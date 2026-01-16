import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Пожалуйста, загрузите фото!")

    is_pro = bool(license_key and len(license_key) > 5)
    
    # ТА САМАЯ РАБОЧАЯ МОДЕЛЬ ОТ LUCATACO (Исправляет 422)
    model_id = "lucataco/interior-design:76604a15c33606f234394622f36f6d3a8258e747ef1f7053e16739665f80b852"
    
    steps = 40 if is_pro else 20

    try:
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 4k, architectural photography",
                "guidance_scale": 7.5,
                "num_inference_steps": steps
            }
        )
        result_url = output[0] if isinstance(output, list) else output
        return result_url, "✨ Готово! Дизайн сгенерирован."
    except Exception as e:
        return None, f"Ошибка: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🏠 AI Interior Designer Pro")
    with gr.Row():
        with gr.Column():
            room_img = gr.Image(type="filepath", label="1. Загрузите фото комнаты")
            style_drop = gr.Dropdown(
                label="2. Выберите стиль",
                choices=["Modern", "Scandinavian", "Luxury", "Minimalist", "Industrial"],
                value="Modern"
            )
            key_in = gr.Textbox(label="3. Код PRO (опционально)", placeholder="Введите любой ключ для 4K")
            btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            result_img = gr.Image(label="Результат")
            status_text = gr.Markdown("Статус: Готов к работе")

    btn.click(generate_design, [room_img, style_drop, key_in], [result_img, status_text])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)
