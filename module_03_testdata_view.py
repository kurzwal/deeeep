import csv

# CSV 파일 경로
# ========= edit this name ========= #
file_name = "pcd_test"
# ================================== #
file_path = "./datasets/processed_data/" + file_name + ".csv"

# 레이블 별 데이터셋 개수를 저장할 딕셔너리
label_counts = {0: 0, 1: 0, 2: 0}

# CSV 파일을 읽어들이며 레이블 별 데이터셋 개수 계산
with open(file_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    
    # header skip
    next(reader)
    
    for row in reader:
        if not row:  # row가 비어있는 경우 skip
            continue
        
        try:
            label = int(row[-1])  # 마지막 column이 레이블로 추정
            if label in label_counts:
                label_counts[label] += 1
        except Exception as e:
            print(f"Error occurred on row: {row}")
            print(f"Error message: {e}")

# 결과 출력
total_datasets = sum(label_counts.values())
print(f"전체 데이터셋 개수 : {total_datasets}")
for label, count in label_counts.items():
    print(f"{label}의 데이터셋 개수 : {count}")
