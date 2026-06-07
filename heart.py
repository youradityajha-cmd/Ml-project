import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 

df=pd.read_csv("C:\\Users\\youra\\Downloads\\heart.csv")
print(df.head())
print(df.columns)
print(df.shape)
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df['HeartDisease'].value_counts().plot(kind='bar'))
plt.show()
plt.figure(figsize=(12,8))

def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
    plt.title(var)

plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)

plt.tight_layout()
plt.show()
    