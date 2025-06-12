import pandas as pd
df = pd.read_parquet('gsm_test.parquet', engine='pyarrow')
print(df.columns.tolist())
