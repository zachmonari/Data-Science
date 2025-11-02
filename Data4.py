import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
#line plot
x=[1,2,3,4,5]
y=[2,4,6,8,10]

plt.figure(figsize=(6,4))
plt.plot(x,y,marker='o')
plt.title('Sales Data')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.grid(alpha=0.3)
plt.show()
"""
#bar chart
df=pd.DataFrame({
    'Name':["Zac","Kev","Dan","Ian","Jack"],
    "Score":[96,80,75,85,90]
})
"""
print(df.describe())
print(df["Score"])
print(df.info())
print(df.shape)
high_scores=df["Score"]>=85
print(high_scores)
sorted1=df.sort_values(by="Score",ascending=False)
print(sorted1)

df["Category"]=pd.cut(
    df["Score"],
    bins=[0,70,80,90,100],
    labels=["D","C","B","A"]
)
print(df)

plt.figure(figsize=(6,4))
sns.barplot(x="Name",y="Score",data=df,color="r")
plt.title("Student Scores")
plt.show()
"""
#histogram
sns.histplot(df["Score"],bins=10,kde=True)
plt.title("Student Scores")
plt.show()
