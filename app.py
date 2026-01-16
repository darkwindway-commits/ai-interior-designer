import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем переменные (токен Replicate берется из настроек Railway)
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Please upload a photo first!")

    # Проверка: если в ключе больше 5 символов — это PRO режим
    is_pro = bool(license_key and len(license_key) > 5)
    
    # Ссылка на модель нейросети
    model_id = "jagadish-b/interior-design:524ca86510e1945391629864a75476a6d68f23f8594244199999999999999999"
    
    # Настройки качества
    steps = 40 if is_pro else 15

    try:
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 4k",
                "num_inference_steps": steps
            }
        )
        status = "✨ PRO Mode Active (High Quality)" if is_pro else "🆓 Free Version (Low Quality)"
        return output[0], status
    except Exception as e:
        return None, f"Error: {str(e)}"

# Создаем интерфейс
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🏠 AI Interior Designer Pro")
    
    with gr.Row():
        with gr.Column():
            room_img = gr.Image(type="filepath", label="1. Upload Your Room")
            style_drop = gr.Dropdown(
                label="2. Choose Style",
                choices=["Modern", "Scandinavian", "Industrial", "Boho", "Minimalist", "Luxury"],
                value="Modern"
            )
            key_in = gr.Textbox(label="3. PRO Access Code", placeholder="Enter your key for 4K quality")
            btn = gr.Button("TRANSFORM ROOM", variant="primary")
        
        with gr.Column():
            result_img = gr.Image(label="Your New Design")
            status_text = gr.Markdown("Status: Ready")

    btn.click(generate_design, [room_img, style_drop, key_in], [result_img, status_text])

    gr.HTML("""
        <div style="text-align: center; background: #f0f7ff; padding: 20px; border-radius: 10px; margin-top: 20px;">
            <h3>Want 4K Quality & No Watermarks?</h3>
            <p>Get 50 high-quality renders for just $9.99</p>
            <a href="https://darkwind4.gumroad.com/l/vmzaq" target="_blank">
                <button style="background: #6366f1; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    Buy PRO License Key
                </button>
            </a>
        </div>
    """)

# Запуск с фиксами для Railway
if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True
    )
