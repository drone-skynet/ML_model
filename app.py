# app.py
# Flask를 통해 API를 제공, 예측 요청을 받아 모델을 통해 결과 반환
from flask import Flask, request, jsonify
from src.model_utils import load_model
import pandas as pd
import os

app = Flask(__name__)

# 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 절대 경로
model_path = os.path.join(current_dir, "models", "delivery_time_prediction_model.pkl")
model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트로부터 JSON 데이터 수신
    data = request.get_json()

    # 입력 데이터 생성
    input_data = pd.DataFrame([data], columns=['distance', 'wind_speed', 'wind_direction', 'drone_direction'])

    # 예측 수행
    prediction = model.predict(input_data)

    # 예측 결과 반환
    return jsonify({'predicted_delivery_time': prediction[0]})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
