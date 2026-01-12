import gradio as gr
import pandas as pd
import os
from .core import detect_objects_stream, set_classes_and_save_model
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

    with gr.Blocks() as demo:
        with gr.Tab("Workspace"):
            model_state = gr.State([])  # Store the model as a state variable
            
            with gr.Row():
                prompts_table = gr.Dataframe(
                    headers=list(prompts_df_new.columns),
                    value=prompts_df_new,
                    interactive=True,
                    label="Class Names (as Column Names) and Descriptions"
                )
            
            with gr.Row():
                with gr.Column():
                    set_classes_button = gr.Button("Prompt Model")
                with gr.Column():
                    model_status = gr.Textbox(
                        label="Status", 
                        value="Model Not Prompted",
                        visible=True
                    )
            
            with gr.Row():
                with gr.Column():
                    gallery = gr.Gallery(
                        label="Upload Images", 
                        show_label=True, 
                        height="auto",
                        type="filepath", 
                        columns=5,
                    )
                with gr.Column():
                    webcam_img = gr.Image(
                        label="Upload Webcam Images",
                        sources=["webcam"]
                    )
    
            with gr.Row():
                batch_slider = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
                btn = gr.Button("Run Detection", variant="primary")
    
            with gr.Row():
                with gr.Column():
                    output_gallery = gr.Gallery(
                        label="Predictions",
                        type="numpy",
                        columns=2
                    )
                with gr.Column():
                    output_table = gr.Dataframe(label="Detection Summary")
                    annotations_folder_state = gr.State([])
    
            # Event handlers
            btn.click(
                fn=detect_objects_stream,
                inputs=[gallery, batch_slider, model_state],
                outputs=[output_gallery, output_table, annotations_folder_state],
            )
            
            webcam_img.change(
                fn=handle_webcam_change_event,
                inputs=[webcam_img, gallery],
                outputs=[gallery]
            )
            
        with gr.Tab("Downloads"):
            # Initial dummy value, assuming 'assets/_' file exists (handled in entry point)
            download_output = gr.File(
                label="Download Model",
                value=os.path.join("assets", "_"),
                interactive=False,
                height="2em",
            )
            
            set_classes_button.click(
                fn=set_classes_and_save_model,
                inputs=[prompts_table],
                outputs=[model_state, download_output, model_status]
            ).then(
                fn=cleanup_temp_model,
                inputs=download_output,
                outputs=None
            )
            
            get_annotations_btn = gr.Button("Get Annotations")
            download_annotations = gr.JSON(
                label="Annotations JSON",
                value=[],
            )
            
            get_annotations_btn.click(
                fn=zip_folder,
                inputs=[annotations_folder_state],
                outputs=[download_annotations]
            )
            
    return demo
