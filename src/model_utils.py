# model_utils.py
# 모델을 저장하거나 로드하는 기능을 캡슐화하여, 다른 파일에서 재사용 가능
import joblib

def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path):
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model
