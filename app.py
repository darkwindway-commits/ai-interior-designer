import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Функция теперь принимает ВСЕ 3 поля из интерфейса
def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Пожалуйста, сначала загрузите фото!")

    # Стабильная модель
    model_id = "adirik/interior-design:76604a15c33606f234394622f36f6d3a8258e747ef1f7053e16739665f80b852"
    
    # Качество зависит от наличия ключа
    is_pro = bool(license_key and len(license_key) > 5)
    steps = 50 if is_pro else 25

    try:
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, photorealistic, 4k",
                "guidance_scale": 9,
                "num_inference_steps": steps
            }
        )
        res = output[0] if isinstance(output, list) else output
        return res, "✨ Готово!"
    except Exception as e:
        return None, f"Ошибка: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            room_img = gr.Image(type="filepath", label="1. Фото комнаты")
            style_drop = gr.Dropdown(choices=["Modern", "Scandinavian", "Luxury"], value="Modern", label="2. Стиль")
            key_in = gr.Textbox(label="3. Код PRO (опционально)") # Третий аргумент
            btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        with gr.Column():
            result_img = gr.Image(label="Ваш новый интерьер")
            status_text = gr.Markdown("Статус: Ожидание")

    # Входы: [Картинка, Стиль, Ключ] -> Выходы: [Картинка, Текст]
    btn.click(generate_design, [room_img, style_drop, key_in], [result_img, status_text])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
