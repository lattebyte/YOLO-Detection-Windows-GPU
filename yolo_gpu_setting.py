import torch
from ultralytics import YOLO
def configure_yolo(model_name):

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load a model
    model = YOLO(model_name+'.pt').to(device)  # Load pre-trained model
    return model