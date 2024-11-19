# predict.py
# 저장된 모델을 로드하고, 새로운 데이터로 배송 소요 시간을 예측
import pandas as pd
import math
from model_utils import load_model
import os

# 효율 점수 계산 함수
def calculate_efficiency_score(drone_direction, wind_direction, wind_speed):
    # 상대 각도 계산
    relative_angle = (wind_direction - drone_direction + 360) % 360
    relative_angle = math.radians(relative_angle)  # 라디안 변환

    # 효율 점수 계산
    efficiency_score = -math.cos(relative_angle) * wind_speed
    return efficiency_score

def preprocess_input_data(input_data):
    """
    입력 데이터를 전처리하여 필요한 피처를 생성.
    """
    # 효율 점수 계산
    input_data['efficiency_score'] = input_data.apply(
        lambda row: calculate_efficiency_score(
            row['drone_direction'], row['wind_direction'], row['wind_speed']
        ),
        axis=1
    )

    # 필요한 피처만 선택
    processed_data = input_data[['distance', 'efficiency_score']]
    return processed_data

def predict_delivery_time(model_file, input_data):
    # 모델 로드
    model = load_model(model_file)

    # 입력 데이터 전처리
    processed_data = preprocess_input_data(input_data)

    # 예측 수행
    predictions = model.predict(processed_data)
    return predictions

if __name__ == "__main__":
    # 현재 스크립트 경로 기반으로 모델 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../models/delivery_time_prediction_model.pkl")

    # 새로운 데이터 예시
    new_data = pd.DataFrame(
        [[10.0, 3.0, 70.0, 90.0], [15.0, 5.0, 120.0, 180.0]],  # 거리, 풍속, 풍향, 드론 방향
        columns=['distance', 'wind_speed', 'wind_direction', 'drone_direction']
    )

    # 예측 실행
    result = predict_delivery_time(model_path, new_data)

    # 결과 출력
    new_data['predicted_delivery_time'] = result
    print(new_data)
