from ultralytics import YOLO
import yaml
import requests
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Update data.yaml with correct paths
data_config = {
    'train': 'train/images',
    'val': 'valid/images', 
    'test': 'test/images',
    'nc': 6,
    'names': ['Normal_Eyes', 'Normal_Mouth', 'SlightPalsy_Eyes', 'SlightPalsy_Mouth', 'StrongPalsy_Eyes', 'StrongPalsy_Mouth']
}

with open('data.yaml', 'w') as f:
    yaml.dump(data_config, f)

# xAI-assisted data validation
def validate_annotations(api_key):
    sample_images = list(Path('train/images').glob('*.jpg'))[:5]
    for img in sample_images:
        response = requests.post('https://api.x.ai/v1/chat/completions',
            headers={'Authorization': f'Bearer {api_key}'},
            json={
                'model': 'grok-vision-beta',
                'messages': [{'role': 'user', 'content': f'Validate face palsy annotations in {img}'}]
            })
        print(f'{img}: {response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")}')

# Train with preprocessing augmentations and early stopping
model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    patience=15,
    save_period=10,
    augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    val=True,
    plots=True
)

# Load best model for inference
model = YOLO('runs/detect/train/best.pt')

# Path to results.csv (update if your run folder is different)
results_path = Path('runs/detect/train/results.csv')
if results_path.exists():
    df = pd.read_csv(results_path)
    metrics = ['train/box_loss', 'train/cls_loss', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP_0.5(B)']

    plt.figure(figsize=(12, 8))
    for metric in metrics:
        if metric in df.columns:
            plt.plot(df['epoch'], df[metric], label=metric)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"Could not find {results_path}. Training graphs will not be plotted.")