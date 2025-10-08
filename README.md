# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, split into features and target, preprocess numerical and categorical data.
2. Create a pipeline combining preprocessing and Decision Tree Classifier.
3. Use GridSearchCV to optimize hyperparameters and train the best model. 
4. Evaluate accuracy, classification metrics, confusion matrix, ROC curve, and visualize the decision tree. 

## Program:

```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: CJ ROHIT
RegisterNumber:  25003117



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, RocCurveDisplay
)

try:
    df = pd.read_csv("employee_churn.csv")
    print("Dataset loaded successfully!\n")
except FileNotFoundError:
    np.random.seed(42)
    df = pd.DataFrame({
        "Age": np.random.randint(22, 60, 300),
        "Department": np.random.choice(["Sales", "HR", "IT", "Finance"], 300),
        "Salary": np.random.choice(["Low", "Medium", "High"], 300),
        "Tenure": np.random.randint(1, 15, 300),
        "Satisfaction": np.round(np.random.rand(300), 2),
        "Churn": np.random.choice([0, 1], 300, p=[0.7, 0.3])
    })
    print("No dataset found. Using synthetic sample.\n")

print(df.head())

X = df.drop("Churn", axis=1)
y = df["Churn"]

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

dt = DecisionTreeClassifier(random_state=42)

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", dt)
])

param_grid = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [3, 5, 7, None],
    "classifier__min_samples_split": [2, 5, 10]
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)

print("\nBest Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)

best_model = grid.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Stay","Churn"], yticklabels=["Stay","Churn"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

if hasattr(best_model.named_steps["classifier"], "predict_proba"):
    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.show()
    auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:,1])
    print("ROC AUC:", auc)

final_tree = best_model.named_steps["classifier"]

ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([numeric_features, cat_names])

plt.figure(figsize=(14,8))
plot_tree(final_tree, feature_names=feature_names, class_names=["Stay","Churn"], filled=True, fontsize=8)
plt.title("Decision Tree - Employee Churn")
plt.show()

```
## Output:
<img width="725" height="187" alt="image" src="https://github.com/user-attachments/assets/049e794f-ecc1-42ab-a651-973416398ec7" />

<img width="1508" height="142" alt="image" src="https://github.com/user-attachments/assets/576ccf4d-c906-4dee-830f-050aeca1682a" />

<img width="938" height="265" alt="image" src="https://github.com/user-attachments/assets/e96eebdd-fd42-49af-b0d1-0c9167888ea1" />

<img width="764" height="429" alt="image" src="https://github.com/user-attachments/assets/2881e008-0968-41ce-98d7-92c1785ccb9b" />

<img width="652" height="483" alt="image" src="https://github.com/user-attachments/assets/359a2875-75e3-4779-be05-d49a76edd665" />

<img width="1575" height="884" alt="image" src="https://github.com/user-attachments/assets/a4df2585-e9f2-420c-a221-6f7659c2e319" />


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
