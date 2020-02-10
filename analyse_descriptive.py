import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/kidney.csv')

data.head(5)

data.shape

data.describe()

plt.hist(data['age'], bins=20)
plt.show()

plt.hist(data['bp'], bins=20)
plt.show()
data['bp'].unique()

plt.hist(data['sg'], bins=10)
plt.show()
data['sg'].unique()

plt.hist(data['al'], bins=10)
plt.show()
data['al'].unique()

plt.hist(data['su'], bins=20)
plt.show()
data['su'].unique()

plt.hist(data['bgr'], bins=30)
plt.show()

plt.hist(data['bu'], bins=30)
plt.show()

plt.hist(data['sc'], bins=30)
plt.show()
data['sc'].unique()

plt.hist(data['sod'], bins=30)
plt.show()

plt.hist(data['pot'], bins=30)
plt.show()
data['pot'].unique()

plt.hist(data['hemo'], bins=20)
plt.show()
data['hemo'].unique()

plt.hist(data['pcv'], bins=30)
plt.show()

plt.hist(data['wbcc'], bins=30)
plt.show()

plt.hist(data['rbcc'], bins=30)
plt.show()
