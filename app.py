from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)
model = YOLO('best.pt')

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
    app.run(host='0.0.0.0', port=5000)