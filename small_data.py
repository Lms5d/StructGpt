import json

# 定义输入文件路径
input_path = './data/tabfact/tab_fact_test.json'

# 读取原始数据
with open(input_path, "rb") as f:
    all_data = json.load(f)

# 提取前500条数据
subset_data = all_data[:200]

# 定义输出文件路径
output_path = input_path.replace("tab_fact_test.json", "test_table.json")

# 保存为新的 JSON 文件
with open(output_path, "w") as f:
    json.dump(subset_data, f, indent=4)  # 使用 indent=4 来格式化 JSON

print(f"Successfully saved the first 200 test examples to {output_path}")
