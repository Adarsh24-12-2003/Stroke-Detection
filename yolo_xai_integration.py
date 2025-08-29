from ultralytics import YOLO
import requests
import json

class FacePalsyAnalyzer:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        self.xai_api = "https://api.x.ai/v1/chat/completions"  # Replace with actual endpoint
        
    def analyze_image(self, image_path, api_key):
        # YOLO detection
        results = self.model(image_path)
        detections = []
        
        for r in results:
            for box in r.boxes:
                detections.append({
                    'class': r.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist()
                })
        
        # xAI interpretation
        prompt = f"Analyze face palsy severity from detections: {json.dumps(detections)}"
        
        response = requests.post(self.xai_api, 
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}]
            })
        
        return {
            'detections': detections,
            'analysis': response.json()['choices'][0]['message']['content']
        }

# Usage
analyzer = FacePalsyAnalyzer()
result = analyzer.analyze_image('test_image.jpg', 'your_xai_api_key')
print(result['analysis'])