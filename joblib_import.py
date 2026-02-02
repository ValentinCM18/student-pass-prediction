import joblib
import pandas as pd

pipeline = joblib.load("/student_model.pkl")

new_student = pd.DataFrame(
    [[18,11]],
    columns=["age", "average"]
)

print(pipeline.predict(new_student))
