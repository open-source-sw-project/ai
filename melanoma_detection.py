import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from efficientnet_pytorch import EfficientNet
import albumentations
import pretrainedmodels
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

class MelanomaModel(nn.Module):
    def __init__(self, model_name='efficientnet-b5', pretrained=True):
        super(MelanomaModel, self).__init__()
        self.model = EfficientNet.from_pretrained(model_name)
        in_features = self.model._fc.in_features
        self.model._fc = nn.Linear(in_features, 1)
        
    def forward(self, x):
        x = self.model(x)
        return x

def preprocess_image(image_path, image_size=640):
    print(f"Preprocessing image: {image_path}")
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = np.array(image)
    augmented = transforms_val(image=image)
    image = augmented['image']
    image = torch.tensor(image).float().permute(2, 0, 1).unsqueeze(0)
    return image

def predict_melanoma(image_path, model_path, device='cuda'):
    print("Initializing model...")
    model = MelanomaModel('efficientnet-b5')
    print(f"Loading weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    print("Preprocessing image...")
    image = preprocess_image(image_path)
    image = image.to(device)
    
    print("Making prediction...")
    with torch.no_grad():
        prediction = torch.sigmoid(model(image))
        
    return prediction.item()

def save_result_report(image_path, probability, output_dir='results'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(image_path)
    risk_level = "고위험" if probability > 0.5 else "저위험"
    
    report_content = f"""
피부 병변 분석 보고서
===========================
검사 일시: {now.strftime("%Y년 %m월 %d일 %H시 %M분")}
분석 이미지: {image_name}

[분석 결과]
악성 흑색종 확률: {probability*100:.2f}%
위험도 판정: {risk_level}

[판정 기준]
- 50% 미만: 저위험
- 50% 이상: 고위험

[주의사항]
1. 이 결과는 AI 분석 결과이며, 참고용으로만 사용해야 합니다.
2. 정확한 진단을 위해서는 반드시 전문의와 상담하시기 바랍니다.
3. 피부의 변화가 있을 경우 정기적인 검진을 권장합니다.

[위험도 상세 설명]
{risk_level}({probability*100:.2f}%) 수준으로 판정되었습니다.
{"전문의의 정밀검진이 필요한 수준입니다." if probability > 0.5 else "정기적인 관찰이 권장됩니다."}
""".strip()

    output_file = os.path.join(output_dir, f'피부분석결과_{timestamp}.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    return output_file

def select_image():
    """파일 선택 대화상자를 열어 이미지 파일을 선택합니다."""
    root = tk.Tk()
    root.withdraw()  # Tkinter 창 숨기기
    
    file_path = filedialog.askopenfilename(
        title='분석할 이미지 파일을 선택하세요',
        filetypes=[
            ('이미지 파일', '*.jpg *.jpeg *.png *.bmp'),
            ('모든 파일', '*.*')
        ]
    )
    
    return file_path if file_path else None

def main():
    # 모델 경로 설정
    model_path = r'weights/4c_b5ns_1.5e_640_ext_15ep_best_fold0.pth'
    
    # 사용자에게 이미지 파일 선택하도록 함
    print("\n피부 병변 이미지 파일을 선택해주세요.")
    image_path = select_image()
    
    if not image_path:
        print("파일이 선택되지 않았습니다. 프로그램을 종료합니다.")
        return
    
    print(f"\n선택된 이미지: {image_path}")
    
    # GPU 사용 가능 여부 확인
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 중인 디바이스: {device}")
    
    # 파일 존재 여부 확인
    if not os.path.exists(model_path):
        print(f"Error: 모델 파일을 찾을 수 없습니다: {os.path.abspath(model_path)}")
        return
    if not os.path.exists(image_path):
        print(f"Error: 이미지 파일을 찾을 수 없습니다: {image_path}")
        return
        
    print("\n분석을 시작합니다...")
    print(f"모델 로딩 중: {model_path}")
    print(f"이미지 처리 중: {image_path}")
    
    try:
        # 예측 실행
        probability = predict_melanoma(image_path, model_path, device)
        print(f'\n[분석 결과]')
        print(f'악성 흑색종 확률: {probability:.4f}')
        risk_level = "고위험" if probability > 0.5 else "저위험"
        print(f'위험도 판정: {risk_level}')
        
        # 결과 저장
        report_file = save_result_report(image_path, probability)
        print(f'\n분석 보고서가 저장되었습니다: {report_file}')
        
        # 계속할지 물어보기
        while True:
            response = input('\n다른 이미지를 분석하시겠습니까? (y/n): ').lower()
            if response in ['y', 'n']:
                if response == 'y':
                    print('\n' + '='*50)
                    main()  # 재귀적으로 다시 실행
                break
            print("'y' 또는 'n'을 입력해주세요.")
        
    except Exception as e:
        print(f"\n에러 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    try:
        print("\n=== 피부 병변 분석 프로그램 ===")
        print("이 프로그램은 피부 병변 이미지를 분석하여 악성 흑색종의 위험도를 평가합니다.")
        print("주의: 이 분석 결과는 참고용이며, 정확한 진단을 위해서는 반드시 전문의와 상담하시기 바랍니다.")
        print("=" * 50)
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    finally:
        print("\n프로그램을 종료합니다.")