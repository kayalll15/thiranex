import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("train.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ── TASK 1: DATA CLEANING ──────────────────────────────

# 1. Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 2. Fill missing Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# 3. Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# 4. Drop Cabin (too many missing)
df.drop(columns=['Cabin'], inplace=True)

# 5. Remove duplicates
df.drop_duplicates(inplace=True)

print("\nAfter Cleaning - Missing Values:")
print(df.isnull().sum())

# ── TASK 1: VISUALIZATIONS ─────────────────────────────

# Chart 1 - Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df, palette='Set2')
plt.title('Survival Count (0=No, 1=Yes)')
plt.savefig('chart1_survival.png')
plt.show()

# Chart 2 - Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df['Age'], bins=30, kde=True, color='steelblue')
plt.title('Age Distribution')
plt.savefig('chart2_age.png')
plt.show()

# Chart 3 - Survival by Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df, palette='Set1')
plt.title('Survival by Gender')
plt.savefig('chart3_gender.png')
plt.show()

# Chart 4 - Survival by Passenger Class
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Passenger Class')
plt.savefig('chart4_pclass.png')
plt.show()

# Chart 5 - Fare Distribution
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Fare', data=df, palette='coolwarm')
plt.title('Fare vs Survival')
plt.savefig('chart5_fare.png')
plt.show()

print("\n✅ Task 1 Complete! All charts saved.")