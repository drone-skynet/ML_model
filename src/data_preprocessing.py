# data_preprocessing.py
# 원시 데이터(raw_data)를 읽어와 전처리 작업을 수행(파생 피처 생성)
# 드론 방향과 풍향을 바탕으로 효율 점수를 계산하여 추가
# 결과를 processed_data.csv로 저장
import pandas as pd
import math
import os

# 효율 점수 계산 함수
def calculate_efficiency_score(drone_direction, wind_direction, wind_speed):
    # 상대 각도 계산
    relative_angle = (wind_direction - drone_direction + 360) % 360
    relative_angle = math.radians(relative_angle)  # 라디안으로 변환

    # 효율 점수 계산
    efficiency_score = -math.cos(relative_angle) * wind_speed
    return efficiency_score

def preprocess_data(input_file, output_file):
    # 원시 데이터 로드
    data = pd.read_csv(input_file)

    # 효율 점수 계산
    data['efficiency_score'] = data.apply(
        lambda row: calculate_efficiency_score(
            row['drone_direction'], row['wind_direction'], row['wind_speed']
        ),
        axis=1
    )

    # 필요한 열만 선택
    processed_data = data[['distance', 'efficiency_score', 'delivery_time']]

    # 전처리된 데이터 저장
    processed_data.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    # 현재 스크립트 위치를 기준으로 파일 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(base_dir, "../data/raw_data.csv")
    processed_data_path = os.path.join(base_dir, "../data/processed_data.csv")
    
    # 전처리 실행
    preprocess_data(raw_data_path, processed_data_path)
