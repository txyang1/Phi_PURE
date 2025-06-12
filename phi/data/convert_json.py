import pandas as pd

# 1. 从 JSON 文件读取
#    如果你的 JSON 文件是一个标准的列表格式（[ {...}, {...}, … ]），用下面这一行：
df = pd.read_json('gsm_test.json')

#    如果是 NDJSON（每行一个 JSON 对象），则：
# df = pd.read_json('data/math_test.json', lines=True)

# 2. 写出为 Parquet
df.to_parquet('gsm_test.parquet', engine='pyarrow', index=False)



