import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── LOAD & CLEAN DATA ──────────────────────────────────
df = pd.read_csv("train.csv")

df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# encode categorical columns
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ── STATISTICAL SUMMARY ────────────────────────────────
print("Dataset Shape:", df.shape)
print("\nStatistical Summary:")
print(df.describe())
print("\nValue Counts - Survived:")
print(df['Survived'].value_counts())

# ── CHART 1: Correlation Heatmap ───────────────────────
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart10_correlation.png')
plt.show()

# ── CHART 2: Age vs Fare Scatterplot ───────────────────
plt.figure(figsize=(7,5))
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df, palette='Set1')
plt.title('Age vs Fare (Colored by Survival)')
plt.tight_layout()
plt.savefig('chart11_scatter.png')
plt.show()

# ── CHART 3: Survival Rate by Embarked ─────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='Embarked', hue='Survived', data=df, palette='Set2')
plt.title('Survival by Embarked Port')
plt.tight_layout()
plt.savefig('chart12_embarked.png')
plt.show()

# ── CHART 4: Age vs Survival Boxplot ───────────────────
plt.figure(figsize=(6,4))
sns.boxplot(x='Survived', y='Age', data=df, palette='coolwarm')
plt.title('Age Distribution by Survival')
plt.tight_layout()
plt.savefig('chart13_agebox.png')
plt.show()

# ── CHART 5: Pairplot ──────────────────────────────────
sns.pairplot(df[['Age', 'Fare', 'Pclass', 'Survived']], hue='Survived', palette='Set1')
plt.suptitle('Pairplot of Key Features', y=1.02)
plt.savefig('chart14_pairplot.png')
plt.show()

print("\n✅ Task 3 Complete! All charts saved.")