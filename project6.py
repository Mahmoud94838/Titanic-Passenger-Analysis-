# Task1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task2
df = sns.load_dataset("titanic")
print(df.head())

# Task3
print(df.head(10))

# Task4
print(df.tail(10))

# Task5
print(df.info())

# Task6
print(df.describe())

# Task7
print("Shape:", df.shape)
print("\nColumns:", df.columns)
print("\nData Types:\n", df.dtypes)

# Task8
print(df['who'])

# Task9
print(df[['sex', 'age', 'survived']])

# Task10
print(df.iloc[30:61])

# Task11
print(df[df['age'] > 30])

# Task12
print(df[df['sex'] == 'female'])

# Task13
print(df[df['survived'] == 1])

# Task14
print(df[df['pclass'] == 1])

# Task15
print(df[df['fare'] > 100])

# Task16
print(df[(df['age'] >= 20) & (df['age'] <= 40)])

# Task17
print(df[(df['sex'] == 'female') & (df['survived'] == 1)])

# Task18
df['Family_Size'] = df['sibsp'] + df['parch']
print(df[['sibsp', 'parch', 'Family_Size']])

# Task19
fare_array = np.array(df['fare'])

print("Maximum:", fare_array.max())
print("Minimum:", fare_array.min())
print("Mean:", fare_array.mean())
print("Sum:", fare_array.sum())

# Task20
print(df['age'].groupby(df['pclass']).mean())

# Task21
print(df['fare'].groupby(df['pclass']).mean())

# Task22
print(df['fare'].groupby(df['pclass']).max())

# Task23
print(df['age'].groupby(df['pclass']).min())

# Task24
print(df['survived'].groupby(df['pclass']).count())

# Task25
avg_per_class = df['fare'].groupby(df['pclass']).mean()

plt.bar(avg_per_class.index, avg_per_class.values, color=['blue', 'red', 'green'])
plt.title("Average")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")
plt.show()

# Task26
plt.hist(df['age'].dropna(), bins=20, color='yellow', edgecolor='black')
plt.title("Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Task27
sns.scatterplot(data=df, x='age', y='fare')
plt.title("Age vs Fare")
plt.show()

# Task28
sns.barplot(x='pclass', y='fare', data=df)
plt.title("Average per Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Average Fare")
plt.show()

# Task29
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution with KDE")
plt.show()

# Task30
sns.histplot(df['age'], bins=20, kde=True)
plt.title("Age Distribution with KDE")
plt.savefig('titanic_histogram.png', dpi=300)
plt.show()