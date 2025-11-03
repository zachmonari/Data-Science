import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

#bar chart
df=pd.DataFrame({
    'Name':["Zac","Kev","Dan","Ian","Jack"],
    "Score":[96,80,75,85,90]
})

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

#histogram
sns.histplot(df["Score"],bins=10,kde=True)
plt.title("Student Scores")
plt.show()

#boxplot
sns.boxplot(data=df,x="Name",y="Score")
plt.title("Student Scores Distribution")
plt.show()

marks=pd.DataFrame({
    "Hours":[2,3,4,5,6,7,8],
    "Score": [55, 58, 63, 70, 72, 80, 85]
})

# scatter plot
sns.scatterplot(data=marks,x="Hours",y="Score")
sns.scatterplot(data=marks,x="Hours",y="Score")
plt.title("Hours studied vs Student Scores")
plt.show()

#regression fit
sns.regplot(data=marks,x="Hours",y="Score")
plt.title("Trend: Hours Studied vs Score")
plt.show()
# correlation

corr=marks["Hours"].corr(marks["Score"])
print(corr)
sns.heatmap(marks,annot=True,cmap="vlag")
plt.title("Correlation Heatmap")
plt.show()

# Full Example
students1=pd.DataFrame({
    "Name":["Zac","Kev","Dan","Ian","Jack"],
    "Maths":[85, 42, 77, 90, 66],
    "English": [88, 55, 79, 92, 70]
})
# Reshape to long for grouped bars
long_df = students1.melt(id_vars="Name", var_name="Subject", value_name="Score")
sns.barplot(x="Name",y="Score",hue="Subject",data=long_df)
plt.title("Student Scores")
plt.ylabel("Score (0–100)")
plt.show()
# Compare subjects with scatter
sns.scatterplot(data=students1,x="Maths",y="English")
plt.title("Math vs English Scores")
plt.xlabel("Maths")
plt.ylabel("English")
plt.show()
# Distribution of Math
sns.histplot(students1["Maths"], bins=5, kde=True)
plt.title("Distribution of Math Scores")
plt.show()

# Distribution of English
sns.histplot(students1["English"], bins=5, kde=True)
plt.title("Distribution of English Scores")
plt.show()
# correlation between subjects
corr1 = students1[["Maths", "English"]].corr()
print(corr1)
sns.heatmap(corr1, annot=True, cmap="vlag")
plt.title("Correlation Heatmap")
plt.show()