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

def predict_delivery_time(model_file, input_data):
    # 모델 로드
    model = load_model(model_file)
    
    # 효율 점수 계산 및 추가
    input_data['efficiency_score'] = input_data.apply(
        lambda row: calculate_efficiency_score(row['drone_direction'], row['wind_direction'], row['wind_speed']),
        axis=1
    )
    
    # 예측 수행
    predictions = model.predict(input_data[['distance', 'efficiency_score']])  # 필요한 열만 사용
    return predictions

if __name__ == "__main__":
    # 현재 스크립트 경로 기반으로 모델 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "delivery_time_prediction_model.pkl")

    # 새로운 데이터 예시
    new_data = pd.DataFrame(
        [[2.0, 5.0, 70.0, 90.0]],  # 거리, 풍속, 풍향, 드론 방향
        columns=['distance', 'wind_speed', 'wind_direction', 'drone_direction']
    )

    # 예측 실행
    result = predict_delivery_time(model_path, new_data)
    print("Predicted Delivery Time (minutes): ", result)
