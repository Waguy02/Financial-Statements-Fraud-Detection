import os

from researchpkg.anomaly_detection.models.financial.lgbm_classifier import (
    LgbmClassifier,
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
        "learning_rate": 0.07991666834823204,
        "max_depth": 7.0,
        "num_estimators": 200,
        "num_leaves": 5,
        "standardize": True,
    }

    model = LgbmClassifier(features_type=CONFIG["features_type"], config=CONFIG)
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
