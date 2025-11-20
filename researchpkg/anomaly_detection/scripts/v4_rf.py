import os

from researchpkg.anomaly_detection.models.financial.rf_classifier import (
    RF_Classifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)

if __name__ == "__main__":

    CONFIG = {
        "fold_id": int(os.environ["FOLD_ID"]),
        "dataset_version": "company_isolated_splitting",
        "decision_threshold": 0.5,
        "features_type": NumericalFeaturesType.EXTENDED,
        "max_depth": 75.0,
        "num_estimators": 100,
        "num_leaves": 20,
        "standardize": False,
    }

    model = RF_Classifier(features_type=CONFIG["features_type"], config=CONFIG)
    model.setup_model()

    train_path, test_path = load_cross_validation_path(CONFIG)
    data = model.load_data(train_path=train_path, test_path=test_path)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = data

    print("Training RF model with PyTorch Lightning...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    print("Evaluating RF model on test set...")
    metrics_test = model.evaluate(
        X_test_scaled, y_test, subset="test", save_metrics=True
    )
