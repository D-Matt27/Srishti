import pandas as pd
import numpy as np
data = {"Name": ["Asha", "Rahul", "Neha"],"Marks": [85, 90, 78]}
df = pd.DataFrame(data)
print(df)

data_dict = {
'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Eva'],
'Age': [25, np.nan, 30, 22, 28, 28],
'Score': [85, 90, 95, 80, 70, 70]
}
df = pd.DataFrame(data_dict)
print(df)
print("Null values per column:\n", df.isnull().sum())