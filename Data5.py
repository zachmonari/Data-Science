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
