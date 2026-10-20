from pathlib import Path

from src.churn_pipeline import load_data, split_features


DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def test_dataset_contract():
    frame = load_data(DATA_PATH)
    features, target, customer_ids = split_features(frame)
    assert len(frame) == 7043
    assert len(features) == len(target) == len(customer_ids)
    assert set(target.unique()) == {0, 1}
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns

