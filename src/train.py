import json
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_config(path="config/config.json"):
    with open(path, "r") as f:
        return json.load(f)

def train_model():
    config = load_config()
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model trained with accuracy: {acc}")
    joblib.dump(model, "model_train.pkl")

if __name__ == "__main__":
    train_model()
