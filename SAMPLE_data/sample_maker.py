import pandas as pd
import os
file_path = 'data/'
file_list = ["sections.csv", "tas.csv", "test1.csv", "test2.csv", "test3.csv"]
for file in file_list:
    df = pd.read_csv(os.path.join(file_path, file))
    df.head(10).to_csv(f"sample_{file}", index=False)
    print(f"Saved sample_{file}")
