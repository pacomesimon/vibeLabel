import os
import datetime
import tempfile
import torch
import numpy as np
import pandas as pd
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
    # Save the model relative to current working directory (usually root)
    model_instance.save(filename)
    return filename

def set_classes_and_save_model(df):
    model_instance = set_classes_with_descriptions(df)
    status = f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}: Model Prompted'
    return model_instance, save_model(model_instance), status
