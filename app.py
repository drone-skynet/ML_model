# app.py
# Flask를 통해 API를 제공, 예측 요청을 받아 모델을 통해 결과 반환
from flask import Flask, request, jsonify
from src.model_utils import load_model
import pandas as pd
import os
import math

app = Flask(__name__)

# 모델 로드
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 절대 경로
model_path = os.path.join(current_dir, "models", "delivery_time_prediction_model.pkl")
model = load_model(model_path)

# 효율 점수 계산 함수
def calculate_efficiency_score(drone_direction, wind_direction, wind_speed):
    # 상대 각도 계산
    relative_angle = (wind_direction - drone_direction + 360) % 360
    relative_angle = math.radians(relative_angle)  # 라디안으로 변환

    # 효율 점수 계산
    efficiency_score = -math.cos(relative_angle) * wind_speed
    return efficiency_score

# 입력 데이터 전처리 함수
def preprocess_input_data(data):
    # 데이터프레임 생성
    input_data = pd.DataFrame(data, columns=['distance', 'wind_direction', 'drone_direction', 'wind_speed'])

    # 효율 점수 계산
    input_data['efficiency_score'] = input_data.apply(
        lambda row: calculate_efficiency_score(
            row['drone_direction'], row['wind_direction'], row['wind_speed']
        ),
        axis=1
    )

    # 필요한 열만 선택
    processed_data = input_data[['distance', 'efficiency_score']]
    return processed_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트로부터 JSON 데이터 수신
        data = request.get_json()

        # 다중 입력 데이터 처리
        if not isinstance(data, list):  # 단일 데이터가 전달된 경우 리스트로 변환
            data = [data]

        # 입력 데이터 전처리
        processed_data = preprocess_input_data(data)

        # 예측 수행
        predictions = model.predict(processed_data)

        # 입력 데이터와 예측 결과를 결합하여 반환
        response = pd.DataFrame(data)
        response['predicted_delivery_time'] = predictions
        return jsonify(response.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
