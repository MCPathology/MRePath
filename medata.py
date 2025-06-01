import pandas as pd
import os

# 定义路径
csv_path = 'datasets_csv/metadata/tcga_blca.csv'
data_dir = '../data/pt_files/'

# 读取CSV文件
df = pd.read_csv(csv_path)

# 检查每个slide_id对应的文件是否存在
def file_exists(slide_id):
    file_path = os.path.join(data_dir, f"{slide_id[:-4]}.pt")
    print(file_path)
    return os.path.isfile(file_path)

# 保留找到的行，删除找不到的行
df_filtered = df[df['slide_id'].apply(file_exists)]

# 保存过滤后的CSV文件（可以覆盖原始文件或另存为新文件）
df_filtered.to_csv(csv_path, index=False)

print("CSV 文件已更新，未找到对应文件的行已删除。")