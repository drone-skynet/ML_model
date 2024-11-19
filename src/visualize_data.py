# visualize_data.py
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# 데이터 로드 함수
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(base_dir, "../data/processed_data.csv")
    return pd.read_csv(processed_data_path)

# 모델 로드 함수
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "../models/delivery_time_prediction_model.pkl")
    return joblib.load(model_path)

# 1. 입력 데이터 분포 시각화 (KDE 그래프 사용)
def visualize_data_distribution(data):
    plt.figure(figsize=(12, 6))

    # Distance의 KDE 시각화
    plt.subplot(1, 2, 1)
    data['distance'].plot(kind='kde', color='skyblue', linewidth=2)
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.title('KDE of Distance')

    # Efficiency Score의 KDE 시각화
    plt.subplot(1, 2, 2)
    data['efficiency_score'].plot(kind='kde', color='orange', linewidth=2)
    plt.xlabel('Efficiency Score')
    plt.ylabel('Density')
    plt.title('KDE of Efficiency Score')

    plt.tight_layout()
    plt.show()

# 2. 모델 성능 평가 시각화 (실제값 vs. 예측값)
def visualize_model_performance(data, model):
    X = data[['distance', 'efficiency_score']]
    y_true = data['delivery_time']
    y_pred = model.predict(X)

    # 실제값 vs. 예측값
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Fit')
    plt.xlabel('Actual Delivery Time')
    plt.ylabel('Predicted Delivery Time')
    plt.title('Actual vs. Predicted Delivery Time')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 데이터 및 모델 로드
    data = load_data()
    model = load_model()

    # 시각화 실행
    visualize_data_distribution(data)       # 데이터 분포 시각화 (KDE)
    visualize_model_performance(data, model)  # 모델 성능 시각화
