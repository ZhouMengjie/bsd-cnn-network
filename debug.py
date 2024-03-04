import os
import csv

def count_zeros_in_column(csv_file, column_index):
    zeros_count = 0

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                value = float(row[column_index])  # assuming the column contains numerical values
                if value == 1:
                    zeros_count += 1
            except ValueError:
                pass  # skip non-numeric values

    return zeros_count

# Example usage
# front, right, back left
csv_file_path = os.path.join('csv','wallstreet5k.csv')  # replace 'example.csv' with your CSV file path
column_index = 1+3  # replace 2 with the index of the column you want to analyze (0-indexed)
zeros_count = count_zeros_in_column(csv_file_path, column_index)
print(f"Number of zeros in column {column_index}: {zeros_count}")
