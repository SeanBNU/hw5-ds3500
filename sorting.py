import pandas as pd

sections_path='data/sections.csv'
tas_path='data/tas.csv'
#ta_cols = ["max_assigned"] + [str(i) for i in range(17)]
tas = (pd.read_csv(tas_path)).iloc[:,3:]
print((pd.read_csv('data/test1.csv', header=None)).values.shape)