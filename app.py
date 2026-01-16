import gradio as gr
import replicate
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
GUMROAD_TOKEN = os.getenv("GUMROAD_TOKEN")
PRODUCT_ID = os.getenv("GUMROAD_PRODUCT_ID")

# Your Gumroad product link
GUMROAD_URL = "https://darkwind4.gumroad.com/l/daesag"

free_usage_tracker = {}

def verify_and_generate(image, style, license_key, request: gr.Request):
    client_ip = request.client.host
    
    # 1. Trial Logic
    if not license_key:
        user_free_count = free_usage_tracker.get(client_ip, 0)
        if user_free_count >= 2:
            raise gr.Error("Your 2 free trials have ended. Please purchase a license key to continue!")
        free_usage_tracker[client_ip] = user_free_count + 1
    
    # 2. License Verification
    else:
        try:
            response = requests.post(
                "https://api.gumroad.com/v2/licenses/verify",
                data={
                    "product_id": PRODUCT_ID,
                    "license_key": license_key,
                    "increment_uses_count": "true"
                }
            )
            data = response.json()
            if not data.get("success"):
                raise gr.Error("Access Error: " + data.get("message", "Invalid key"))
        except Exception as e:
            raise gr.Error("Verification failed. Please check your internet connection.")

    if not image:
        raise gr.Error("Please upload a photo first!")

    # 3. AI Generation (Improved prompt for structure)
    try:
        # We use a prompt that emphasizes keeping the original architecture
        prompt = f"A professional {style} interior design, photographic, highly detailed. Maintain the original room structure, walls, and window positions from the photo. 8k resolution."
        
        output = replicate.run(
            "stability-ai/stable-diffusion-3.5-large",
            input={
                "prompt": prompt,
                "image": open(image, "rb"),
                "aspect_ratio": "1:1",
                "output_format": "jpg",
                "cfg": 4.5
            }
        )
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return output
    except Exception as e:
        raise gr.Error("AI Generation Error. Please try again.")

# UI with Gumroad Link
with gr.Blocks(theme=gr.themes.Soft(), title="AI Interior Designer") as demo:
    gr.Markdown("# 🏠 AI Interior Designer")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Upload Room Photo")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Luxury", "Minimalist", "Industrial", "Boho"], 
                value="Modern", 
                label="2. Select Style"
            )
            key_input = gr.Textbox(label="3. License Key", placeholder="Enter key here", type="password")
            
            run_btn = gr.Button("RE-DESIGN MY ROOM ✨", variant="primary")
            
            # Button to Gumroad
            gr.Markdown("---")
            gr.Markdown("### 💎 Get more generations")
            gr.Button("🛒 BUY LICENSE KEY (50 DESIGNS)", link=GUMROAD_URL)
            
        with gr.Column():
            output_img = gr.Image(label="Result")
            gr.Markdown("ℹ️ *Free version: 2 trials per user. For unlimited structure control, use a license key.*")

    run_btn.click(fn=verify_and_generate, inputs=[input_img, style_drop, key_input], outputs=output_img)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
