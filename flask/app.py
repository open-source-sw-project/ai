from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np
import albumentations
import os
from datetime import datetime
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import io
import sys

app = Flask(__name__)

class MelanomaModel(nn.Module):
    def __init__(self, model_name='efficientnet-b5', pretrained=True):
        super(MelanomaModel, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        return self.model(x)

def preprocess_image(image_bytes, image_size=640):
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = np.array(image)
    augmented = transforms_val(image=image)
    image = augmented['image']
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

def generate_report(image_name, probability):
    risk_level = "고위험" if probability > 0.5 else "저위험"
    now = datetime.now()
    
    report = {
        "timestamp": now.strftime("%Y년 %m월 %d일 %H시 %M분"),
        "image_name": image_name,
        "melanoma_probability": f"{probability*100:.2f}%",
        "risk_level": risk_level,
        "assessment": "전문의의 정밀검진이 필요한 수준입니다." if probability > 0.5 else "정기적인 관찰이 권장됩니다.",
        "notice": [
            "이 결과는 AI 분석 결과이며, 참고용으로만 사용해야 합니다.",
            "정확한 진단을 위해서는 반드시 전문의와 상담하시기 바랍니다.",
            "피부의 변화가 있을 경우 정기적인 검진을 권장합니다."
        ]
    }
    return report

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Server is running"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': '이미지 파일이 필요합니다.'}), 400
    
    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # 이미지 전처리
        image = preprocess_image(image_bytes)
        image = image.to(device)
        
        # 예측 수행
        with torch.no_grad():
            prediction = torch.sigmoid(model(image))
            probability = prediction.item()
        
        # 결과 리포트 생성
        report = generate_report(image_file.filename, probability)
        
        return jsonify({
            'success': True,
            'report': report
        }), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': '모델 파일을 찾을 수 없습니다.', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': '예측 중 문제가 발생했습니다.', 'details': str(e)}), 500

def load_model(model_path, device):
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully. Using device: {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # 환경 변수에서 모델 경로 설정
    model_path = os.getenv('MODEL_PATH', './model/4c_b5ns_1.5e_640_ext_15ep_best_fold0.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 모델 초기화
    model = MelanomaModel('efficientnet-b5')
    load_model(model_path, device)
    
    # Flask 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=False)
