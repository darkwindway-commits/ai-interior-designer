import gradio as gr
import replicate
import os
from dotenv import load_dotenv

load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    raise ValueError("Add REPLICATE_API_TOKEN to .env or Railway Variables")

os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

def generate_design(image, style, license_key):
    if not image:
        raise gr.Error("Please upload your room photo first!")

    # Проверка ключа (пока простая, позже подключим Gumroad API)
    is_pro = bool(license_key and len(license_key) > 5)  # если ключ введён — PRO

    prompt = f"A professional {style} interior redesign of the uploaded room, high quality, photorealistic, 8k, cinematic lighting, architectural photography, masterpiece"
    negative_prompt = "ugly, blurry, low quality, distorted, watermark, text, messy, dark, bad anatomy"

    steps = 40 if is_pro else 15
    guidance = 7.5 if is_pro else 5.0

    try:
        output = replicate.run(
            "jagadish-b/interior-design:524ca86510e1945391629864a75476a6d68f23f8594244199999999999999999",
            input={
                "image": open(image, "rb"),
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "guidance_scale": guidance
            }
        )
        status = "PRO mode active (high quality)" if is_pro else "Free mode (limited quality)"
        return output[0], status

    except Exception as e:
        raise gr.Error(f"Error: {str(e)}. Try again or buy PRO.")

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as demo:
    gr.Markdown("# 🏠 AI Interior Designer Pro")
    gr.Markdown("Upload a photo of your room → get a stunning redesign! (Real img2img)")

    with gr.Row():
        with gr.Column():
            room_photo = gr.Image(label="Upload Room Photo (required)", type="filepath", height=350)
            style_choice = gr.Dropdown(
                label="Choose Interior Style",
                choices=["Minimalist", "Scandinavian", "Industrial", "Boho", "Modern", "Japanese", "Rustic", "Luxury"],
                value="Modern"
            )
            key_input = gr.Textbox(label="PRO License Key (from Gumroad)", placeholder="Enter key for 4K & unlimited")
            generate_btn = gr.Button("Generate Redesign", variant="primary", size="lg")

        with gr.Column():
            result_img = gr.Image(label="Your New Interior")
            status_msg = gr.Markdown("Status: Free mode (limited quality)")

    generate_btn.click(
        fn=generate_design,
        inputs=[room_photo, style_choice, key_input],
        outputs=[result_img, status_msg]
    )

    gr.HTML("""
    <div style="text-align: center; margin-top: 40px; padding: 25px; background: #f8f9fa; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h3 style="color: #1e40af;">Want Unlimited 4K Generations?</h3>
        <p>Buy PRO Package: 50 high-quality renders, no watermarks</p>
        <a href="https://darkwind4.gumroad.com/l/vmzaq" target="_blank">
            <button style="background: #6366f1; color: white; padding: 16px 40px; border: none; border-radius: 12px; font-size: 20px; font-weight: bold; cursor: pointer; box-shadow: 0 8px 20px rgba(99,102,241,0.4); transition: all 0.3s;">
                Get PRO Key – $9.99
            </button>
        </a>
    </div>
    """)

if __name__ == "__main__":

    demo.launch(server_name="0.0.0.0", server_port=7860)
