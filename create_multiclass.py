import pandas as pd

data = pd.read_csv('data/kidney.csv')

data.head()

data.loc[data['sg'] == 1.005, 'sg'] = 0
data.loc[data['sg'] == 1.01, 'sg'] = 1
data.loc[data['sg'] == 1.015, 'sg'] = 2
data.loc[data['sg'] == 1.02, 'sg'] = 3
data.loc[data['sg'] == 1.025, 'sg'] = 4

data['sg'] = data['sg'].astype(int)

data.dtypes

data.head()

data.to_csv('data/multiclass.csv', index = False)
