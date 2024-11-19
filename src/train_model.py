# train_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가 라이브러리
import joblib
import math
import os

# processed_data.csv 데이터를 읽어 모델을 학습하고 평가
# 학습된 모델을 delivery_time_prediction_model.pkl로 저장하여 flask 서버에서 사용할 수 있도록 준비
def train_model(data_file, model_file):
    # 데이터 읽기
    data = pd.read_csv(data_file)

    # 입력 피처와 타겟 변수 설정
    X = data[['distance', 'efficiency_score']]  # 입력 피처
    y = data['delivery_time']  # 타겟 변수

    # 데이터 분할 (학습용 데이터와 테스트 데이터를 8:2 비율로 분할)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 모델 평가
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차(예측값과 실제값의 차이를 평가)
    r2 = r2_score(y_test, y_pred)  # 모델의 설명력을 나타내며, 1에 가까울수록 좋은 모델

    print("Mean Squared Error (MSE):", round(mse, 2))
    print("R^2 Score:", round(r2, 2))

    # 모델 저장
    os.makedirs(os.path.dirname(model_file), exist_ok=True)  # 모델 저장 디렉토리 생성
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트의 디렉토리
    data_path = os.path.join(base_dir, "../data", "processed_data.csv")
    model_path = os.path.join(base_dir, "../models", "delivery_time_prediction_model.pkl")

    # 모델 학습 및 저장
    train_model(data_path, model_path)
