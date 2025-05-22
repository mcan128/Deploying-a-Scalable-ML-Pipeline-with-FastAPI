import pytest
from ml.model import train_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ml.model import compute_model_metrics
from ml.data import process_data
import pandas as pd

def test_one():
    """
    # Ensure RandomForestClassifier is the correct model
    """
    X_train = np.random.rand(10,5)

    y_train = np.random.randint(0, 2, 10)

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    pass


def test_two():
    """
    # Verify compute_model_metrics returns expected values
    """
    y_true = [1, 0, 1, 1, 0]
    preds = [1, 0, 1, 0, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, preds)
    assert isinstance(precision, float), "Precision should be a float"
    assert isinstance(recall, float), "Recall should be a float"
    assert isinstance(fbeta, float), "F1-Score should be a float"
    pass


def test_three():
    """
    # Verify process_data outputs expected formats
    """
    sample_data = pd.DataFrame({
        "feature1": ["A", "B", "A"],
        "feature2": ["X", "Y", "X"],
        "salary": [1, 0, 1]
    })
    categorical_features = ["feature1", "feature2"]
    X, y, encoder, lb = process_data(sample_data, categorical_features, label="salary", training=True)
    assert X.shape[0] == 3, "X should have 3 rows"
    assert len(y) == 3, "y should have 3 elements"
    pass
