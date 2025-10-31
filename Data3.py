import pandas as pd
#load the data
df=pd.read_csv('student-data.csv')
#display first 5 rows
print(df.head())
#data summary
print(df.info())
#statistical summary
print(df.describe())
#missing values
print(df.isna())

