import joblib
from sklearn.datasets import load_digits

def inference():
    model = joblib.load("model_train.pkl")
    digits = load_digits()
    preds = model.predict(digits.data[:10])
    print("Predictions for first 10 samples:", preds)

if __name__ == "__main__":
    inference()
