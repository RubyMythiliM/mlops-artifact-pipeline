import sys
import os
import joblib
import json

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import load_config, train_model
from sklearn.linear_model import LogisticRegression

def test_config_loading():
    config = load_config()
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_training():
    train_model()
    model = joblib.load("model_train.pkl")
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
