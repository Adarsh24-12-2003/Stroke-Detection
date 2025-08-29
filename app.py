from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load model with ultralytics
from ultralytics import YOLO
import os
os.environ['TORCH_SERIALIZATION_SAFE_GLOBALS'] = 'True'
model = YOLO('best.pt')

class_names = ['Normal_Eyes', 'Normal_Mouth', 'SlightPalsy_Eyes', 'SlightPalsy_Mouth', 'StrongPalsy_Eyes', 'StrongPalsy_Mouth']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file.stream)
        results = model(image)
        
        predictions = []
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    predictions.append({
                        'class': model.names[int(box.cls)],
                        'confidence': float(box.conf),
                        'bbox': box.xyxy[0].tolist()
                    })
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)