import pandas as pd
import numpy as np

#Loading messy data
messy = pd.DataFrame({
    "name": ["John Smith","JANE DOE","bob johnson","Jane Doe","Jane A. Doe"],
    "age":  [25, 999, 30, np.nan, 25],
    "salary": [50000, -10000, 75000, 60000, "$50,000"],
    "date_joined": ["01/15/2020","2021-03-22","March 5, 2019","12-25-2020","01/15/2020"]
})
#Remove exact duplicates
messy=messy.drop_duplicates()
print(messy)
print()
#Standardize text case for names
messy["name"]=messy["name"].str.title()
messy["name"].head()

#Fix clearly impossible ages
messy.loc[messy["age"]>120,"age"]=np.nan
print(messy["age"])

#Make salary numeric and drop negatives
s = messy["salary"].astype(str).str.replace(r"[$,]", "", regex=True)
messy["salary"] = pd.to_numeric(s, errors="coerce")
messy.loc[messy["salary"] < 0, "salary"] = np.nan
print(messy[["salary"]])

#Fill missing ages with the median
if not messy["age"].isna().all():
    messy["age"] = messy["age"].fillna(messy["age"].median())
print(messy["age"])