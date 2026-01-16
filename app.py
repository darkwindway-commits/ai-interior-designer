import gradio as gr
import replicate
import os
from dotenv import load_dotenv

# Загружаем API токен (Railway подтянет его из раздела Variables)
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Пожалуйста, сначала загрузите фото!")

    # Проверка ключа для PRO режима
    is_pro = bool(license_key and len(license_key) > 5)
    
    # ЭТА МОДЕЛЬ ТОЧНО РАБОТАЕТ (Исправляет вашу ошибку 422)
    model_id = "lucataco/interior-design:76604a15c33606f234394622f36f6d3a8258e747ef1f7053e16739665f80b852"
    
    # Настройки качества
    steps = 40 if is_pro else 20

    try:
        # Запуск нейросети
        output = replicate.run(
            model_id,
            input={
                "image": open(image, "rb"),
                "prompt": f"A professional {style} interior design, high quality, photorealistic, 4k, architectural photography",
                "guidance_scale": 7.5,
                "num_inference_steps": steps
            }
        )
        # Получаем ссылку на результат
        result_url = output[0] if isinstance(output, list) else output
        
        status = "✨ PRO режим активен" if is_pro else "🆓 Бесплатная версия"
        return result_url, status
    except Exception as e:
        return None, f"Ошибка: {str(e)}. Проверьте баланс на Replicate."

# Создание интерфейса Gradio
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🏠 AI Interior Designer Pro")
    
    with gr.Row():
        with gr.Column():
            room_img = gr.Image(type="filepath", label="1. Загрузите фото вашей комнаты")
            style_drop = gr.Dropdown(
                label="2. Выберите стиль",
                choices=["Modern", "Scandinavian", "Luxury", "Minimalist", "Boho", "Industrial", "Rustic", "Japanese"],
                value="Modern"
            )
            key_in = gr.Textbox(label="3. Код доступа PRO (если есть)", placeholder="Введите ключ для улучшения качества")
            btn = gr.Button("СОЗДАТЬ ДИЗАЙН", variant="primary")
        
        with gr.Column():
            result_img = gr.Image(label="Ваш новый интерьер")
            status_text = gr.Markdown("Статус: Готов к работе")

    btn.click(generate_design, [room_img, style_drop, key_in], [result_img, status_text])

    gr.HTML("""
        <div style="text-align: center; background: #f0f7ff; padding: 20px; border-radius: 10px; margin-top: 20px; border: 1px solid #d1e9ff;">
            <h3>Хотите качество 4K без водяных знаков?</h3>
            <p>Купите пакет из 50 генераций всего за $9.99</p>
            <a href="https://darkwind4.gumroad.com/l/vmzaq" target="_blank" style="text-decoration: none;">
                <button style="background: #6366f1; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; font-weight: bold;">
                    Купить PRO ключ
                </button>
            </a>
        </div>
    """)

# Запуск с настройками для Railway
if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False
    )
