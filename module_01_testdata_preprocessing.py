import csv

def preprocess_data_from_files(filenames, output_filename):
    all_features = []
    all_labels = []

    for filename in filenames:
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # skip the header

            data = [row for row in reader]

            for i in range(0, len(data) - 260, 20):  # Assuming there are more than 260 rows
                feature_block = data[i:i+240]
                label_block = data[i+240:i+260]

                if len(feature_block) < 240 or len(label_block) < 20:
                    # If data is insufficient in the current file, we skip processing this block
                    continue

                feature_vector = []
                for row in feature_block:
                    # ignoring open_time and open columns
                    high = float(row[2])
                    low = float(row[3])
                    open_price = float(row[1])
                    high_normalized = high / open_price * 100
                    low_normalized = low / open_price * 100
                    close_normalized = float(row[4]) / open_price * 100
                    quote_volume = float(row[7])
                    count = int(row[8])
                    taker_buy_quote_volume = float(row[10])

                    feature_vector.extend([high_normalized, low_normalized, close_normalized, quote_volume, count, taker_buy_quote_volume])

                # Convert feature_vector to string
                feature_string = str(feature_vector)

                all_features.append(feature_string)

                # Calculate label
                first_open = float(label_block[0][1])
                label = 1
                for row in label_block:
                    high_normalized = float(row[2]) / first_open * 100
                    low_normalized = float(row[3]) / first_open * 100
                    if high_normalized >= 101 and low_normalized <= 99:
                        label = 1
                        break
                    elif high_normalized >= 101:
                        label = 2
                        break
                    elif low_normalized <= 99:
                        label = 0
                        break
                all_labels.append(label)

    # Write to output file
    with open(output_filename, 'w') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["features", "label"])  # header
        for f, l in zip(all_features, all_labels):
            writer.writerow([f, l])

# File names to process
file_name_first = "./datasets/raw_data/"
file_extension = ".csv"

file_names = [
    file_name_first + "BTCUSDT-1m-2023-07" + file_extension,
]

pcd_name = "pcd_test"

preprocess_data_from_files(file_names, "./datasets/processed_data/" + pcd_name + ".csv")
