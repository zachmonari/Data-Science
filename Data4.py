import matplotlib.pyplot as plt

x=[1,2,3,4,5]
y=[2,4,6,8,10]

plt.figure(figsize=(6,4))
plt.plot(x,y,marker='o')
plt.title('Sales Data')
plt.xlabel('Days')
plt.ylabel('Sales')
plt.grid(alpha=0.3)
plt.show()