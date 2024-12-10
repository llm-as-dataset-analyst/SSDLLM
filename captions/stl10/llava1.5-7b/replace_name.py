import csv

# 读取地点编号和名称的对应关系
def read_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 2:
                # 去除前缀并转换为整数
                key = int(parts[1])
                value = parts[0].split('/')[-1]
                mapping[key] = value
    return mapping

# 替换 CSV 文件中的数字编号为地点名称
def replace_numbers_in_csv(csv_path, mapping):
    with open(csv_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        rows = list(reader)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for row in rows:
            # 替换数字编号为地点名称
            if row[0].isdigit():
                row[0] = mapping[int(row[0])]
            writer.writerow(row)

# 主函数
def main():
    mapping_file = 'classes_name.txt'  # 替换为实际的文件路径
    csv_file = 'stl10_caption_llava1.5-7b.csv'  # 替换为实际的 CSV 文件路径

    mapping = read_mapping(mapping_file)
    replace_numbers_in_csv(csv_file, mapping)

if __name__ == '__main__':
    main()