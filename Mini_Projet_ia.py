from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Charger les données
df = pd.read_csv("/home/cvm/python_project/Projets/students_clean.csv")

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
print("Prediction du nouveau élève: ", prediction)