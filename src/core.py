import os
import datetime
import tempfile
import torch
import numpy as np
import pandas as pd
import json
import google.generativeai as genai
from PIL import Image
from ultralytics import YOLO
from .utils import batch_iterable

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, 'assets')
# Construct the absolute path to the model file
MODEL_PATH = os.path.join(ASSETS_DIR, 'yoloe-11s-seg.pt')

# Initialize default model globally (or placeholder)
try:
    # Ensure usage of the moved model file
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Warning: Failed to load model from {MODEL_PATH}. Error: {e}")
    model = None

def set_classes_with_descriptions(class_description_dict):
    """
    Sets the class names and aggregated prompt embeddings for the model.
    """
    # Load model from assets
    local_model = YOLO(MODEL_PATH)
    
    class_description_dict = dict(class_description_dict)
    names = tuple(class_description_dict.keys())
    all_pe_s_list = []
    
    for name in names:
        descriptions = tuple(class_description_dict[name])
        pe_s = local_model.get_text_pe(descriptions)
        pe_s_aggregated = pe_s.mean(dim=1, keepdim=True)
        all_pe_s_list.append(pe_s_aggregated[0])
        
    pe_s_aggregated = torch.cat(all_pe_s_list, dim=0).unsqueeze(0)
    local_model.set_classes(names, pe_s_aggregated)
    return local_model


def detect_objects_stream(images, batch_size=3, model_instance=None):
    """
    Run YOLO on images in batches, stream each batch's results.
    """
    # Use global model if none provided, though usually passed from state
    if model_instance is None:
        model_instance = model
        
    if not images:
        yield [], pd.DataFrame([{"Error": "No images uploaded"}]), None
        return
        
    if isinstance(model_instance, list) or model_instance is None:
        yield [], pd.DataFrame([{"Error": "Model is not Prompted"}]), None
        return

    all_counts = {}
    all_results = []
    names = model_instance.names

    # Use temp directory for annotations
    folder_name = tempfile.mkdtemp()
    
    for batch in batch_iterable(images, batch_size):
        # Gradio Gallery with type='filepath' returns list of items.
        # Original code assumed list of tuples/list: [img[0] for img in batch]
        # We preserve this logic.
        batch_paths = [img[0] for img in batch]
        np_batch = [Image.open(img).convert('RGB') for img in batch_paths]

        # Run YOLO prediction
        results = model_instance(np_batch, verbose=False, conf=0.01)

        for res_id, res in enumerate(results):
            # Save annotation to txt
            annotation_filename = os.path.basename(batch_paths[res_id])
            annotation_filename = os.path.splitext(annotation_filename)[0]
            output_path = os.path.join(folder_name, f"{annotation_filename}.txt")
            res.save_txt(output_path)
            
            # Plot
            plotted_rgb = res.plot()
            all_results.append(plotted_rgb)

            # Count classes
            if res.boxes and res.boxes.cls is not None:
                for c in res.boxes.cls.cpu().numpy():
                    class_name = names[int(c)]
                    all_counts[class_name] = all_counts.get(class_name, 0) + 1

        # Create DataFrame
        data = {"number_of_images": [len(all_results)]}
        for class_name in names:
            data[class_name] = [all_counts.get(class_name, 0)]
        df = pd.DataFrame(data)

        # Stream partial result
        yield all_results, df, folder_name


def save_model(model_instance):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"yoloe-11s-seg__{timestamp}__.pt"
    # Save the model to a temporary directory
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)
    model_instance.save(filepath)
    return filepath

def set_classes_and_save_model(df):
    model_instance = set_classes_with_descriptions(df)
    status = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model Prompted'
    return model_instance, save_model(model_instance), status

def refine_prompts_with_gemini(prompts_df, output_images):
    """
    Refines prompts using Gemini based on previous prompts and detection results.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        # If no API key is provided, we can't refine.
        # In a real app, we might want to alert the user.
        print("Warning: GEMINI_API_KEY not found. Please set it in environment variables.")
        return prompts_df

    genai.configure(api_key=api_key)

    # Convert prompts_df to dict
    # Each column is a class, values are prompts
    prompts_dict = prompts_df.to_dict(orient='list')
    # Clean up NaNs which happen in Dataframes when columns have different lengths
    cleaned_prompts = {k: [v for v in l if pd.notna(v) and v != ""] for k, l in prompts_dict.items()}

    # Create the function declaration for Gemini
    properties = {
        k: {
            "type": "string", 
            "description": f"A better, more descriptive prompt for the class '{k}' to improve detection accuracy."
        } for k in cleaned_prompts.keys()
    }
    
    declaration = {
        "name": "update_prompts",
        "description": "Updates the prompts for target classes based on visual feedback from previous detections.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(cleaned_prompts.keys())
        }
    }

    model = genai.GenerativeModel(
        model_name="gemini-3-flash-preview",
        tools=[{"function_declarations": [declaration]}]
    )

    # Prepare visual context
    contents = [
        "I am working on an object detection task. Here are the current prompts/descriptions for each class:",
        json.dumps(cleaned_prompts, indent=2),
        "Based on the provided images (which show the current detection results with bounding boxes), "
        "please refine the prompts to be more specific and helpful for the detection model. "
        "Use the 'update_prompts' function to return the new prompts."
    ]

    if output_images:
        for item in output_images:
            # Gradio Gallery returns a list of items. 
            # If type='numpy', it should be a numpy array or a tuple (array, label)
            img_data = item
            if isinstance(item, (list, tuple)):
                img_data = item[0]
            
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'))
                contents.append(img)
            elif isinstance(img_data, str) and os.path.exists(img_data):
                img = Image.open(img_data)
                contents.append(img)

    # Call Gemini
    try:
        response = model.generate_content(contents)
        
        # Extract the function call
        call = response.candidates[0].content.parts[0].function_call
        if call:
            new_prompts_map = dict(call.args)
            # Reconstruct the DataFrame
            # Gradio Dataframe expects a format matching the headers.
            # We'll create a single-row DataFrame with the new prompts.
            # If the user wants multiple prompts, Gemini would have to provide a list, 
            # but the instruction said "each parameter will be a string".
            new_df_data = {k: [new_prompts_map.get(k, "")] for k in prompts_df.columns}
            return pd.DataFrame(new_df_data)
    except Exception as e:
        print(f"Error during Gemini refinement: {e}")
        return prompts_df

    return prompts_df
