import os

from researchpkg.anomaly_detection.models.financial.mlp_classifier import (
    MLP_Classifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)

if __name__ == "__main__":

    CONFIG = {
        "model_name": "mlp_classifier",
        "fold_id": int(os.environ["FOLD_ID"]),
        "batch_size": 64,
        "dataset_version": "company_isolated_splitting",
        "decision_threshold": 0.5,
        "dropout_rate": 0.45907717851305374,
        "epochs": 2,
        "features_type": NumericalFeaturesType.DECHOW,
        "hidden_dims": [],
        "learning_rate": 0.1,
        "oversample": True,
        "patience": 20,
        "standardize": True,
    }

    model = MLP_Classifier(features_type=CONFIG["features_type"], config=CONFIG)

    train_path, test_path = load_cross_validation_path(CONFIG)
    data = model.load_data(train_path=train_path, test_path=test_path)
    X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test = data

    print("Training MLP model with PyTorch Lightning...")
    model.fit(X_train_scaled, y_train, X_val_scaled, y_val)

    print("Evaluating MLP model on test set...")
    metrics_test = model.evaluate(
        X_test_scaled, y_test, subset="test", save_metrics=True
    )
