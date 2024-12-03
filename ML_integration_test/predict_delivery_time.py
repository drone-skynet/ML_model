from ml_model import load_model, predict_delivery_time

# 학습된 모델 로드
model = load_model("./delivery_time_prediction_model.pkl")

# 인자 값
path = {
    'distance': 1.5765170374447863, # db값 이용
    'wind_speed': 3.5, # 기상청 API 아용
    'wind_direction': 180.2, # 기상청 API 아용
    'drone_direction': 90.7 # 역과 역 사이의 각도로 각색 필요
}

# 배달 시간 예측
predicted_time = predict_delivery_time(
    model,
    distance=path['distance'],
    wind_speed=path['wind_speed'],
    wind_direction=path['wind_direction'],
    drone_direction=path['drone_direction']
)

print(f"예측된 배달 시간: {predicted_time:.2f} 분")
