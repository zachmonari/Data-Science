import pandas as pd
import matplotlib.pyplot as plt

# Example dataset
data = {'gender': ['Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Male']}
df = pd.DataFrame(data)

# Count how many of each gender
gender_counts = df['gender'].value_counts()

# Plot pie chart
plt.figure(figsize=(5,5))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'pink'])
plt.title('Gender Distribution')
plt.show()
