import gradio as gr
import pandas as pd
import os
from .core import detect_objects_stream, set_classes_and_save_model, refine_prompts_with_gemini
from .utils import zip_folder

# Default prompts
prompts_df_new = pd.DataFrame({
    "eat": ["mouth", "nose"],
    "see": ["eyes", "head"],
})

def handle_webcam_change_event(webcam_img, gallery):
    """
    Update gallery with webcam image.
    """
    if webcam_img is None:
        webcam_img = []
    else:
        webcam_img = [webcam_img]
    if gallery is None:
        gallery = []
    return gallery + webcam_img

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

    with gr.Blocks(title="VibeLabel - Multi-Modal Labeling") as demo:
        model_state = gr.State([])  # Store the model as a state variable
        annotations_folder_state = gr.State([])

        with gr.Row():
            # --- SIDEBAR (Brain) ---
            with gr.Column(scale=1, min_width=300):
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
            with gr.Column(scale=3):
                gr.Markdown("### 📸 Input Sources")
                with gr.Row():
                    with gr.Column():
                        gallery = gr.Gallery(
                            label="Image Gallery", 
                            show_label=False, 
                            height="300px",
                            type="filepath", 
                            columns=4,
                        )
                    with gr.Column():
                        webcam_img = gr.Image(
                            label="Webcam Snap",
                            sources=["webcam"],
                            height="300px"
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
