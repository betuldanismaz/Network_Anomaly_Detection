import pandas as pd
import sys

file_path = sys.argv[1]

nrows = int(sys.argv[2]) if len(sys.argv) > 2 else 100

df = pd.read_csv(file_path, nrows=nrows)
print(f"\n📊 First {nrows} rows of {file_path}")
print(f"Total columns: {len(df.columns)}")
print(f"\n{df.head(20)}")
print(f"\n{df.info()}")