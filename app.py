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

# Temporary in-memory storage for free trials (clears on restart)
free_usage_tracker = {}

def verify_and_generate(image, style, license_key, request: gr.Request):
    # Get user IP for rate limiting
    client_ip = request.client.host
    
    # 1. FREE TRIAL LOGIC (If no key provided)
    if not license_key:
        user_free_count = free_usage_tracker.get(client_ip, 0)
        
        if user_free_count >= 2:
            raise gr.Error("Your 2 free trials have ended. Please purchase a license key to continue!")
        
        # Log usage
        free_usage_tracker[client_ip] = user_free_count + 1
        print(f"IP {client_ip}: used free trial {user_free_count + 1}/2")
    
    # 2. PAID LICENSE LOGIC
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
                error_msg = data.get("message", "Invalid license key")
                raise gr.Error(f"Access Error: {error_msg}")
        except Exception as e:
            if "Access Error" in str(e): raise e
            raise gr.Error(f"Payment verification failed: {str(e)}")

    # 3. IMAGE VALIDATION
    if not image:
        raise gr.Error("Please upload a photo of your room first!")

    # 4. AI GENERATION (Stable Diffusion 3.5 Large)
    try:
        model_id = "stability-ai/stable-diffusion-3.5-large"
        prompt = f"A professional {style} interior design, high quality, photorealistic, architectural photography, 8k"
        
        output = replicate.run(
            model_id,
            input={
                "prompt": prompt,
                "aspect_ratio": "1:1",
                "output_format": "jpg",
                "cfg": 4.5
            }
        )
        
        # Handle Replicate output correctly to avoid FileNotFoundError
        if isinstance(output, list) and len(output) > 0:
            return output[0]
        return output
        
    except Exception as e:
        raise gr.Error(f"AI Generation failed: {str(e)}")

# Gradio Interface in English
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏠 AI Interior Designer")
    gr.Markdown("🌟 **Special Offer:** 2 trials for free! For a full package (50 generations), please enter your license key.")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="filepath", label="1. Upload Room Photo")
            style_drop = gr.Dropdown(
                choices=["Modern", "Scandinavian", "Luxury", "Minimalist", "Industrial", "Boho"], 
                value="Modern", 
                label="2. Select Design Style"
            )
            key_input = gr.Textbox(
                label="3. License Key", 
                placeholder="Leave empty for free trial",
                type="password"
            )
            run_btn = gr.Button("GENERATE DESIGN ✨", variant="primary")
        
        with gr.Column():
            output_img = gr.Image(label="Your New Interior")

    # Link button to function
    run_btn.click(
        fn=verify_and_generate, 
        inputs=[input_img, style_drop, key_input], 
        outputs=output_img
    )

# Launch app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
