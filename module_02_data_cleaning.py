import csv
import random

# 원본 CSV 파일 경로
# ========= edit this name ========= #
sourtce_name = "pcd_all"
destination_name = "cleaned_pcd_all"
# ================================== #

source_path = "./datasets/processed_data/" + sourtce_name + ".csv"
# 정제된 데이터를 저장할 CSV 파일 경로
destination_path = "./datasets/processed_data/" + destination_name + ".csv"

# 데이터를 저장할 리스트
minus_one_data = []
zero_data = []
one_data = []

# 문제가 되는 행의 개수를 카운트하는 변수 추가
problematic_rows_count = 0

# CSV 파일을 읽어들이며 레이블 별로 데이터 분류
with open(source_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    # header 추출
    header = next(reader)
    
    for row in reader:
        # 행이 비어있거나 예상보다 짧으면 카운트 후 스킵
        if not row or len(row) < 2:
            problematic_rows_count += 1
            continue

        try:
            label = int(row[-1])  # 마지막 column이 레이블로 추정
        except ValueError:  # 숫자로 변환할 수 없는 값이 있을 경우 카운트 후 스킵
            problematic_rows_count += 1
            continue
        
        if label == 0:
            minus_one_data.append(row)
        elif label == 1:
            zero_data.append(row)
        elif label == 2:
            one_data.append(row)

# -1과 1의 데이터 개수를 최대한 보존하면서 0의 데이터 개수를 줄임
# -1과 1의 데이터 개수를 합한 만큼 0의 데이터 개수를 필터링
num_samples_for_zero = int((len(minus_one_data) + len(one_data)) / 3 * 2)
balanced_zero_data = random.sample(zero_data, num_samples_for_zero)

# CSV 파일로 저장
with open(destination_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # 헤더 쓰기
    writer.writerows(minus_one_data)
    writer.writerows(balanced_zero_data)
    writer.writerows(one_data)

# 문제가 되는 행의 개수 출력
print("문제가 되는 행의 개수:", problematic_rows_count)