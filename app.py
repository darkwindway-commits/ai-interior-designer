import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Подгружаем токен
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Пожалуйста, загрузите фото!")

    # ИСПОЛЬЗУЕМ НОВУЮ СТАБИЛЬНУЮ МОДЕЛЬ
    model_id = "adirik/interior-design:76604a15c33606f234394622f36f6d3a8258e747ef1f7053e16739665f80b852"
    
    try:
        # Запуск с параметрами новой модели
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic",
                "guidance_scale": 9,
                "num_inference_steps": 50
            }
        )
        # Модель возвращает список, берем первый элемент
        return output[0] if isinstance(output, list) else output, "✨ Готово!"
    except Exception as e:
        return None, f"Ошибка: {str(e)}"

# Интерфейс
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Дизайнер Интерьера")
    with gr.Row():
        with gr.Column():
            room_img = gr.Image(type="filepath", label="Фото комнаты")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Industrial", "Minimalist"], 
                value="Modern", 
                label="Стиль"
            )
            btn = gr.Button("ТРАНСФОРМАЦИЯ", variant="primary")
        with gr.Column():
            result_img = gr.Image(label="Результат")
            status_text = gr.Markdown("Статус: Готов")

    btn.click(generate_design, [room_img, style_drop], [result_img, status_text])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
