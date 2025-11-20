import os

from researchpkg.anomaly_detection.models.financial.xgb_classifier import (
    XGB_Classifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)

if __name__ == "__main__":

    CONFIG = {
        "fold_id": int(os.environ["FOLD_ID"]),
        "dataset_version": "company_isolated_splitting",
        "max_depth": 0,
        "num_estimators": 50,
        "num_leaves": 50,
        "decision_threshold": 0.5,
        "learning_rate": 0.05,
        "features_type": NumericalFeaturesType.EXTENDED_DECHOW,
        "standardize": True,
    }

    model = XGB_Classifier(features_type=CONFIG["features_type"], config=CONFIG)
    model.setup_model()

    train_path, test_path = load_cross_validation_path(CONFIG)
    data = model.load_data(train_path=train_path, test_path=test_path)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = data

    print("Training MLP model with PyTorch Lightning...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    print("Evaluating MLP model on test set...")
    metrics_test = model.evaluate(
        X_test_scaled, y_test, subset="test", save_metrics=True
    )
