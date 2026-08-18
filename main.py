import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load Iris Dataset
df = pd.read_csv(r"C:\Users\Administrator\Downloads\Iris.csv")

X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Species"]

# Step 2: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Step 3: Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# Step 4: Evaluate Model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy :", accuracy)

# Step 5: Serialize Model using Joblib
joblib.dump(model, "iris_model.pkl")

print("Model Serialized Successfully")

# Step 6: Load Serialized Model
loaded_model = joblib.load("iris_model.pkl")

print("Serialized Model Loaded Successfully")

# Step 7: Predict New Sample
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = loaded_model.predict(sample)

print("Predicted Flower :", prediction[0])