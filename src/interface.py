import gradio as gr
import pandas as pd
import os
from .core import detect_objects_stream, set_classes_and_save_model, refine_prompts_with_gemini
from .utils import zip_folder

# Define assets path relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'assets', 'images')

# Default gallery images
DEFAULT_IMAGES = []
if os.path.exists(IMAGES_DIR):
    # Gradio Gallery with type='filepath' expects a list of [path, label] or [(path, label)]
    # based on the usage in core.py: batch_paths = [img[0] for img in batch]
    DEFAULT_IMAGES = [[os.path.join(IMAGES_DIR, f), None] for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Default prompts
prompts_df_new = pd.DataFrame({
    "damaged house": ["damaged house", "house with broken roof", "surrounded by debris after a disaster"],
    "non damaged house": ["house", "intact building", "no visible structural damage or debris"],
})

def handle_webcam_change_event(webcam_img, gallery):
    """
    Update gallery with webcam image.
    """
    if webcam_img is None:
        return gallery
    
    # webcam_img is a path if gr.Image(type="filepath")
    new_item = [webcam_img, None]
    
    if gallery is None:
        gallery = []
    
    return gallery + [new_item]

def create_demo():
    
    # Helper for cleanup
    def cleanup_temp_model(file_path):
        if file_path and os.path.exists(file_path):
             try:
                 # Check if it looks like a temp model file just to be safe
                 if "yoloe-11s-seg" in file_path and file_path.endswith(".pt"):
                    os.remove(file_path)
             except Exception as e:
                 print(f"Error removing file: {e}")

    css = """
    .gradio-container {
        background-image: url('https://static.vecteezy.com/system/resources/thumbnails/068/341/035/small_2x/colorful-wavy-line-abstract-background-with-gradient-neon-flow-in-purple-and-blue-on-dark-background-for-futuristic-digital-design-vector.jpg') !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }
    .glass-card {
        background: rgba(15, 15, 25, 0.5) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        padding: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .dark .glass-card {
        background: rgba(10, 10, 20, 0.6) !important;
    }
    /* Enhance text readability on dark background */
    .gradio-container h3, .gradio-container p, .gradio-container span {
        text-shadow: 0px 2px 4px rgba(0,0,0,0.5) !important;
    }
    /* Apply glass to common blocks */
    .gr-form, .gr-box, .gr-panel {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    /* Button enhancements */
    button.primary {
        background: linear-gradient(90deg, #8a2be2 0%, #4b0082 100%) !important;
        border: none !important;
        font-weight: bold !important;
        transition: all 0.3s ease !important;
    }
    button.primary:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
    }
    """

    with gr.Blocks(title="VibeLabel - Multi-Modal Labeling", theme=gr.themes.Soft(), css=css) as demo:
        model_state = gr.State([])  # Store the model as a state variable
        annotations_folder_state = gr.State([])

        with gr.Row():
            # --- SIDEBAR (Brain) ---
            with gr.Column(scale=1, min_width=300, elem_classes=["glass-card"]):
                gr.Markdown("### 🧠 Model Configuration")
                model_status = gr.Textbox(
                    label="Status", 
                    value="Model Not Prompted",
                    interactive=False
                )
                
                prompts_table = gr.Dataframe(
                    headers=list(prompts_df_new.columns),
                    value=prompts_df_new,
                    interactive=True,
                    label="Class Names & Descriptions",
                    wrap=True
                )
                
                with gr.Row():
                    set_classes_button = gr.Button("Prompt Model", variant="secondary")
                    refine_btn = gr.Button("AI Refine", variant="secondary")
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Inference Settings")
                conf_slider = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Confidence Threshold")
                batch_slider = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                
                gr.Markdown("---")
                download_output = gr.File(
                    label="Download Prompted Model (.pt)",
                    value=os.path.join("assets", "_"),
                    interactive=False,
                    height="auto",
                )

            # --- MAIN AREA (Workspace) ---
            with gr.Column(scale=3, elem_classes=["glass-card"]):
                gr.Markdown("### 📸 Input Sources")
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Image Gallery", 
                            show_label=False, 
                            height="300px",
                            type="filepath", 
                            columns=4,
                            value=DEFAULT_IMAGES
                        )
                    with gr.Column():
                        webcam_img = gr.Image(
                            label="Webcam Snap",
                            sources=["webcam"],
                            height="300px",
                            type="filepath"
                        )
                
                btn = gr.Button("🚀 Run Detection", variant="primary", size="lg")
                
                gr.Markdown("### 🎯 Predictions")
                output_gallery = gr.Gallery(
                    label="Detection Results",
                    show_label=False,
                    type="numpy",
                    columns=2,
                    height="auto"
                )
                
                with gr.Accordion("📊 Details & Export", open=False):
                    with gr.Row():
                        with gr.Column(scale=2):
                            output_table = gr.Dataframe(label="Detection Summary")
                        with gr.Column(scale=1):
                            with gr.Row():
                                get_annotations_btn = gr.Button("📦 Package Annotations")
                            with gr.Row():
                                download_annotations = gr.JSON(label="Annotations JSON")

        # --- EVENT HANDLERS ---
        
        # Model Preparation
        set_classes_button.click(
            fn=set_classes_and_save_model,
            inputs=[prompts_table],
            outputs=[model_state, download_output, model_status]
        ).then(
            fn=cleanup_temp_model,
            inputs=download_output,
            outputs=None
        )
        
        refine_btn.click(
            fn=refine_prompts_with_gemini,
            inputs=[prompts_table, output_gallery],
            outputs=[prompts_table]
        )

        # Image Handling
        webcam_img.change(
            fn=handle_webcam_change_event,
            inputs=[webcam_img, gallery],
            outputs=[gallery]
        )

        # Execution
        btn.click(
            fn=detect_objects_stream,
            inputs=[gallery, batch_slider, model_state, conf_slider],
            outputs=[output_gallery, output_table, annotations_folder_state],
        )
        
        # Export
        get_annotations_btn.click(
            fn=zip_folder,
            inputs=[annotations_folder_state],
            outputs=[download_annotations]
        )

    return demo
