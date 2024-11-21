# [Gradient Boosting Regressor 모델]

## 1. 학습 전 수집 데이터
① 경로 거리(km), ② 풍속(m/s), ③ 풍향(°), ④ 드론 경로 방향(°), ⑤ 배송 시간(min)

### (1) 풍향을 고려한 상대 각도 계산
Relative Angle = (Wind Direction - Drone Direction + 360) % 360

0도에 가까울수록 바람이 배송 시간에 더욱 방해

180도에 가까울수록 바람이 배송 시간에 더욱 도움

### (2) cos를 사용한 효율 점수 계산
Relative Angle을 먼저 라디안 형식으로 바꿔준 후 아래의 공식에 대입

Efficiency Score = -cos(Relative Angle) x Wind Speed

여기서 효율 점수의 부호가 바람의 저항(음수)과 도움(양수)를 나타냄

e.g. 효율 점수가 음수라면 바람이 배송 시간에 방해되는 것

## 2. 최종 학습 데이터
X : [경로 거리, 효율 점수]

y : 배송 시간

## 3. 사용 라이브러리
* <b>Scikit-learn</b> - 머신러닝(Gradient Boosting Regressor 모델)
* <b>Pandas</b> - 데이터 분석
* <b>Joblib</b> - 병렬 처리

## 4. 파일 생성 흐름

입력 파일) data/raw_data.csv: 원시 데이터 (수집 데이터)

python src/data_preprocessing.py

생성 결과물) data/processed_data.csv: 전처리된 데이터 (학습 데이터)

python src/train_model.py

생성 결과물) models/delivery_time_prediction_model.pkl (학습된 모델)

## 5. 데이터 시각화 (R^2 = 0.87)

python src/visualize_data.py

<img src="https://github.com/user-attachments/assets/cbb50ca9-8681-473d-ad45-1220b838cb52" width="800" height="400"/>
<img src="https://github.com/user-attachments/assets/e4ca81b5-e36a-4857-80b1-3da6bc478100" width="400" height="400"/>
