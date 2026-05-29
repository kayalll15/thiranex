import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ── LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("diabetes.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nStatistical Summary:")
print(df.describe())

# ── CLEAN DATA ─────────────────────────────────────────
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN using numpy
for col in cols_with_zeros:
    df[col] = df[col].replace(0, float('nan'))
    df[col] = df[col].fillna(df[col].median())

print("\nAfter Cleaning - Missing Values:")
print(df.isnull().sum())

# ── CHART 1: Outcome Count ─────────────────────────────
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df, hue='Outcome', palette='Set2', legend=False)
plt.title('Diabetes Outcome Count (0=No, 1=Yes)')
plt.tight_layout()
plt.savefig('chart15_outcome.png')
plt.show()

# ── CHART 2: Correlation Heatmap ───────────────────────
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart16_correlation.png')
plt.show()

# ── CHART 3: Glucose vs BMI Scatterplot ────────────────
plt.figure(figsize=(7,5))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=df, palette='Set1')
plt.title('Glucose vs BMI (Colored by Outcome)')
plt.tight_layout()
plt.savefig('chart17_scatter.png')
plt.show()

# ── CHART 4: Age Distribution by Outcome ───────────────
plt.figure(figsize=(6,4))
sns.boxplot(x='Outcome', y='Age', data=df, hue='Outcome', palette='coolwarm', legend=False)
plt.title('Age Distribution by Diabetes Outcome')
plt.tight_layout()
plt.savefig('chart18_age.png')
plt.show()

# ── CHART 5: Glucose Distribution ─────────────────────
plt.figure(figsize=(6,4))
sns.histplot(df['Glucose'], bins=30, kde=True, color='steelblue')
plt.title('Glucose Level Distribution')
plt.tight_layout()
plt.savefig('chart19_glucose.png')
plt.show()

# ── ML MODEL ───────────────────────────────────────────
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features].astype(float)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nRandom Forest Accuracy: {rf_acc:.2f}")

# ── CHART 6: Confusion Matrix ──────────────────────────
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Diabetes Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('chart20_confusion.png')
plt.show()

# ── CHART 7: ROC Curve ─────────────────────────────────
plt.figure(figsize=(6,4))
rf_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - Diabetes Prediction')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('chart21_roc.png')
plt.show()

# ── CHART 8: Feature Importance ────────────────────────
plt.figure(figsize=(6,4))
feat_imp = pd.Series(rf.feature_importances_, index=features)
feat_imp.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance - Diabetes Prediction')
plt.tight_layout()
plt.savefig('chart22_features.png')
plt.show()

print("\n✅ Task 4 Complete! All charts saved.")