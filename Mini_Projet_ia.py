from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import joblib

# Charger les données
df = pd.read_csv("students_clean.csv")

df["status"] = np.where(df["average"] >= 10, "PASS", "FAIL")

# Séparer X et Y
x = df[["age", "average"]]
y = df["status"]

# Split Train | Test
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)

# Pipeline (Scaler + Model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

# Entrainer
pipeline.fit(x_train, y_train)

# Accuracy
accuracy = pipeline.score(x_test, y_test)
print("Accuracy: ", accuracy)

# Prédiction d'un nouvel élève
new_student = pd.DataFrame(
    [[21, 19]],
    columns = ["age", "average"]
)

prediction = pipeline.predict(new_student)
probabilities = pipeline.predict_proba(new_student)
print("Prediction du nouveau élève: ", prediction)
print("Probabilité du nouvea élève: ", probabilities)

# Utilisation de joblib pour pas réentrainer a chaque fois
joblib.dump(pipeline, "student_model.pkl")

# Pour recharger plus tard
pipeline = joblib.load("student_model.pkl")

new_student = pd.DataFrame(
    [[20, 15]],
    columns = ["age", "average"]
)

prediction = pipeline.predict(new_student)
print("Prediction: ", prediction)

# Visualisation de la performance du modele avec 'confusion_matrix'
y_pred = pipeline.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Probabilités de prédiction
proba = pipeline.predict_proba(new_student)

print("Probabilité FAIL :", proba[0][0])
print("Probabilité PASS :", proba[0][1])

# L'influence de la prédiction
coefficients = pipeline.named_steps["model"].coef_[0]
features = x.columns

for f, c in zip(features, coefficients):
    print(f"{f} influence: {c:.2f}")
