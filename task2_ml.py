import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ── LOAD DATA ──────────────────────────────────────────
df = pd.read_csv("train.csv")

# ── CLEAN DATA ─────────────────────────────────────────
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# ── ENCODE CATEGORICAL ─────────────────────────────────
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# ── PREPARE FEATURES ───────────────────────────────────
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = df[features]
y = df['Survived']

# verify no NaN
print("NaN in X:", X.isnull().sum().sum())
print("NaN in y:", y.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── TRAIN MODELS ───────────────────────────────────────
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print(f"\nLogistic Regression Accuracy: {lr_acc:.2f}")
print(f"Decision Tree Accuracy:       {dt_acc:.2f}")
print(f"Random Forest Accuracy:       {rf_acc:.2f}")

# ── CHART 1: Accuracy Comparison ───────────────────────
plt.figure(figsize=(6,4))
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracies = [lr_acc, dt_acc, rf_acc]
sns.barplot(x=models, y=accuracies, palette='Set2')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('chart6_accuracy.png')
plt.show()

# ── CHART 2: Confusion Matrix ───────────────────────────
plt.figure(figsize=(6,4))
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('chart7_confusion.png')
plt.show()

# ── CHART 3: ROC Curve ─────────────────────────────────
plt.figure(figsize=(6,4))
rf_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.title('ROC Curve - Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.savefig('chart8_roc.png')
plt.show()

# ── CHART 4: Feature Importance ────────────────────────
plt.figure(figsize=(6,4))
feat_imp = pd.Series(rf.feature_importances_, index=features)
feat_imp.sort_values().plot(kind='barh', color='steelblue')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.savefig('chart9_features.png')
plt.show()

print("\n✅ Task 2 Complete! All charts saved.")