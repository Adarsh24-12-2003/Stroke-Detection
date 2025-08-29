from flask import Flask, request, jsonify
import torch
from PIL import Image
import io
import numpy as np

app = Flask(__name__)

# Load model directly with PyTorch
try:
    model = torch.jit.load('best.pt', map_location='cpu')
    model.eval()
except:
    # Fallback: load as state dict
    model = torch.load('best.pt', map_location='cpu')
    if isinstance(model, dict):
        model = model.get('model', model)

class_names = ['Normal_Eyes', 'Normal_Mouth', 'SlightPalsy_Eyes', 'SlightPalsy_Mouth', 'StrongPalsy_Eyes', 'StrongPalsy_Mouth']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['image']
        image = Image.open(file.stream).convert('RGB')
        
        # Simple preprocessing
        image = image.resize((640, 640))
        img_array = np.array(image) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
        
        # Inference
        with torch.no_grad():
            results = model(img_tensor)
        
        # Simple response
        predictions = [{
            'status': 'Model loaded successfully',
            'image_processed': True,
            'classes_available': class_names
        }]
        
        return jsonify({'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)