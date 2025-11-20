import json
import logging

# Set multiprocessing start method to 'spawn' for compatibilit
import multiprocessing
import os
import tempfile
from pathlib import Path

import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from joblib import Parallel, delayed
from matplotlib.pylab import Enum
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    HPRAM_TUNING_BASE_DIR,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.financial.lgbm_classifier import (
    LgbmClassifier,
)
from researchpkg.anomaly_detection.models.financial.mlp_classifier import (
    MLP_Classifier,
)
from researchpkg.anomaly_detection.models.financial.rf_classifier import (
    RF_Classifier,
)
from researchpkg.anomaly_detection.models.financial.xgb_classifier import (
    XGB_Classifier,
)
from researchpkg.anomaly_detection.models.utils import (
    NumericalFeaturesType,
    load_cross_validation_path,
)
from researchpkg.utils import numpy_to_scalar


class FinancialMOdelType(str, Enum):
    RF = "RF"
    LGBM = "LGBM"
    XGB = "XGB"
    MLP = "MLP"
    LOGISTIC = "LOGISTIC"


DATASET_VERSION = os.environ["DATASET_VERSION"]
assert DATASET_VERSION in [
    "company_isolated_splitting",
    "time_splitting",
], f"Invalid dataset version: {DATASET_VERSION}. Must be one of ['v4', 'v5']"

MAX_EVALS = int(os.environ["MAX_EVALS"])


MODEL_CLASS = os.environ["MODEL_CLASS"]  # Should be one of "RF", "LGBM", "XGB", "MLP"
assert MODEL_CLASS in [
    m.value for m in FinancialMOdelType
], f"Invalid model type: {MODEL_CLASS}. Must be one of {[m.value for m in FinancialMOdelType]}"


FEATURES_TYPE = os.environ[
    "FEATURES_TYPE"
]  # Should be one of "numerical", "categorical", "mixed"
assert FEATURES_TYPE in [
    NumericalFeaturesType.EXTENDED_DECHOW.name,
    NumericalFeaturesType.EXTENDED.name,
    NumericalFeaturesType.DECHOW.name,
], f"Invalid features type: {FEATURES_TYPE}. Must be one of {[NumericalFeaturesType.EXTENDED_DECHOW.name, NumericalFeaturesType.EXTENDED.name, NumericalFeaturesType.DECHOW.name]}"
# Convert to Enum for consistency
FEATURES_TYPE = NumericalFeaturesType(FEATURES_TYPE)


TUNING_DIRS_DICT = {
    FinancialMOdelType.RF: HPRAM_TUNING_BASE_DIR
    / "rf_tuning"
    / f"{DATASET_VERSION}"
    / f"{FEATURES_TYPE.name}",
    FinancialMOdelType.LGBM: HPRAM_TUNING_BASE_DIR
    / "lgbm_tuning"
    / f"{DATASET_VERSION}"
    / f"{FEATURES_TYPE.name}",
    FinancialMOdelType.XGB: HPRAM_TUNING_BASE_DIR
    / "xgb_tuning"
    / f"{DATASET_VERSION}"
    / f"{FEATURES_TYPE.name}",
    FinancialMOdelType.MLP: HPRAM_TUNING_BASE_DIR
    / "mlp_tuning"
    / f"{DATASET_VERSION}"
    / f"{FEATURES_TYPE.name}",
    FinancialMOdelType.LOGISTIC: HPRAM_TUNING_BASE_DIR
    / "logistic_tuning"
    / f"{DATASET_VERSION}"
    / f"{FEATURES_TYPE.name}",
}

MODELS_CLASS_DICT = {
    FinancialMOdelType.RF: RF_Classifier,
    FinancialMOdelType.LGBM: LgbmClassifier,
    FinancialMOdelType.XGB: XGB_Classifier,
    FinancialMOdelType.MLP: MLP_Classifier,  # Assuming MLP_Classifier is defined elsewhere
    FinancialMOdelType.LOGISTIC: MLP_Classifier,  # Assuming Logistic Regression is implemented in MLP_Classifier
}


SEARCH_SPACES_DICTS = {
    FinancialMOdelType.RF: {
        "max_depth": hp.quniform("max_depth", 2, 100, 1),  # Random Forest max depth
        "num_estimators": hp.choice(
            "num_estimators", [50, 100, 200]
        ),  # Number of trees
        "num_leaves": hp.choice("num_leaves", [5, 10, 20, 50]),  # Number of leaves
        "decision_threshold": 0.5,
        "dataset_version": DATASET_VERSION,
        "features_type": FEATURES_TYPE,
        "base_experiment_dir": TUNING_DIRS_DICT[FinancialMOdelType.RF],
        "standardize": hp.choice("standardize", [True, False]),
    },
    FinancialMOdelType.LGBM: {
        "max_depth": hp.quniform("max_depth", 2, 100, 1),
        "num_estimators": hp.choice(
            "num_estimators", [50, 100, 200]
        ),  # Number of trees
        "num_leaves": hp.choice("num_leaves", [5, 10, 20, 50]),  # Number of leaves
        "decision_threshold": 0.5,
        "standardize": hp.choice("standardize", [True, False]),
        "dataset_version": DATASET_VERSION,
        "features_type": FEATURES_TYPE,
        "base_experiment_dir": TUNING_DIRS_DICT[FinancialMOdelType.RF],
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),
    },
    FinancialMOdelType.XGB: {
        "max_depth": hp.quniform("max_depth", 2, 100, 1),
        "num_estimators": hp.choice(
            "num_estimators", [50, 100, 200]
        ),  # Number of trees
        "num_leaves": hp.choice("num_leaves", [5, 10, 20, 50]),  # Number of leaves
        "decision_threshold": 0.5,
        "standardize": hp.choice("standardize", [True, False]),
        "features_type": FEATURES_TYPE,
        "base_experiment_dir": TUNING_DIRS_DICT[FinancialMOdelType.RF],
        "dataset_version": DATASET_VERSION,
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.1),  # Learning rate
    },
    FinancialMOdelType.MLP: {
        "hidden_dims": hp.choice(
            "hidden_dims",
            [
                [32],
                [64],
                [128],
                [512],
                [32, 32],
                [64, 64],
                [128, 128],
                [512, 512],
                [32, 64],
                [64, 128],
                [128, 256],
                [256, 512],
                [512, 1024],
            ],
        ),  # Hidden layer dimensions
        "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),  # Dropout rate
        "learning_rate": hp.choice(
            "learning_rate", [0.001, 0.01, 0.1]
        ),  # Learning rate
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),  # Batch size
        "decision_threshold": 0.5,
        "standardize": hp.choice("standardize", [True, False]),
        "features_type": FEATURES_TYPE,
        "base_experiment_dir": TUNING_DIRS_DICT[FinancialMOdelType.RF],
        "dataset_version": DATASET_VERSION,
        "oversample": hp.choice("oversample", [True, False]),
        "epochs": 2,
        "patience": 20,
    },
    FinancialMOdelType.LOGISTIC: {
        "hidden_dims": hp.choice(
            "hidden_dims",
            [
                [],  # Linear model (Logistic Regression)
            ],
        ),  # Hidden layer dimensions
        "dropout_rate": hp.uniform("dropout_rate", 0.0, 0.5),  # Dropout rate
        "learning_rate": hp.choice(
            "learning_rate", [0.001, 0.01, 0.1]
        ),  # Learning rate
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),  # Batch size
        "decision_threshold": 0.5,
        "standardize": hp.choice("standardize", [True, False]),
        "features_type": FEATURES_TYPE,
        "base_experiment_dir": TUNING_DIRS_DICT[FinancialMOdelType.RF],
        "dataset_version": DATASET_VERSION,
        "oversample": hp.choice("oversample", [True, False]),
        "epochs": 2,
        "patience": 20,
    },
}


TUNING_DIR = TUNING_DIRS_DICT[MODEL_CLASS]


def _train_single_fold(
    model_type: FinancialMOdelType,
    X_train,
    y_train,
    X_val,
    y_val,
    config: dict,
):

    config["base_experiment_dir"] = TUNING_DIRS_DICT[model_type]
    model_class = MODELS_CLASS_DICT[model_type]
    model = model_class(features_type=FEATURES_TYPE, config=config)
    model.fit(X_train, y_train, X_val, y_val)
    eval_metrics = model.evaluate(X_val, y_val, subset="val")
    return eval_metrics["auc_score"]


def objective(
    params: dict,
    model_type: FinancialMOdelType,
    data_by_fold_scaled: dict,
    data_by_fold_unscaled: dict,
) -> dict:
    """
    Objective function for Hyperopt optimization. Evaluates hyperparameters across multiple folds.

    Args:
        params: Parameters to evaluate for the current trial.
        data_by_fold: Dictionary containing (X_train, y_train, X_val, y_val) for each fold.

    Returns:
        dict: Results including loss (negative mean AUC) and status for Hyperopt.
    """
    logging.info(f"Evaluating parameters: {params}")

    data_by_fold = (
        data_by_fold_scaled if params["standardize"] else data_by_fold_unscaled
    )

    # Use joblib.Parallel for multiprocessing across folds
    auc_scores = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(_train_single_fold)(
            model_type,
            X_train,
            y_train,
            X_val,
            y_val,
            {
                **params,
                "fold_id": fold_id,
            },
        )
        for fold_id, (X_train, y_train, X_val, y_val, _, _) in data_by_fold.items()
    )

    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    logging.info(f"Mean AUC for current trial: {mean_auc:.4f}")

    return {
        "loss": -mean_auc,  # Hyperopt minimizeReturns, so negative AUC [2]
        "auc": mean_auc,
        "params": params,
        "status": STATUS_OK,
    }


def load_data_for_fold(fold_id: int, standardize: bool) -> tuple:
    """
    Load data for a specific fold.

    Args:
        fold_id: The ID of the fold to load data for.
        standardize: Whether to standardize the data.

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    config = {
        "fold_id": fold_id,
        "dataset_version": DATASET_VERSION,
        "standardize": standardize,
        "hidden_dims": [],
        "decision_threshold": 0.5,
        "features_type": FEATURES_TYPE,
        "dropout_rate": 0.2,
        "learning_rate": 0.001,
        "batch_size": 1,
        "epochs": 1,
        "patience": 1,
    }

    train_path, test_path = load_cross_validation_path(config)

    model_class = MODELS_CLASS_DICT[FinancialMOdelType.MLP]
    model = model_class(features_type=FEATURES_TYPE, config=config)

    return model.load_data(train_path, test_path)


def run_hyperparams_tuning() -> Trials:
    """
    Run hyperparameter tuning using Hyperopt.

    Args:
        params_space: Hyperparameter search space.
        model_type: Type of financial model to tune.
        data_by_fold: Dictionary containing (X_train, y_train, X_val, y_val) for each fold.

    Returns:
        Trials object containing results of the optimization.
    """

    params_space = SEARCH_SPACES_DICTS[MODEL_CLASS]

    model_type = FinancialMOdelType(MODEL_CLASS)

    scaled_data_by_fold = {}
    unscaled_data_by_fold = {}

    # for fold_id in tqdm(range(1, 6), desc="Loading data for each fold"):
    #     dummy_config = {
    #         "model_name": "mlp_classifier",
    #         "features_type": FEATURES_TYPE,
    #         "hidden_dims": [],
    #         "dropout_rate": 0.2,
    #         "learning_rate": 0.001,
    #         "batch_size": 256,
    #         "epochs": 10,
    #         "patience": 5,
    #         "decision_threshold": 0.5,
    #         "fold_id": fold_id,
    #         "dataset_version": DATASET_VERSION,
    #         "base_experiment_dir": "/tmp/financial_model_tuning",
    #         "oversample": False,
    #     }

    #     dummy_model_unscaled = {**dummy_config, "standardize": False}
    #     dummy_model_scaled = {**dummy_config, "standardize": True}

    #     train_path, test_path = load_cross_validation_path(dummy_config)

    #     scaled_data_by_fold[fold_id] = MLP_Classifier(
    #         features_type=FEATURES_TYPE, config=dummy_model_scaled
    #     ).load_data(train_path, test_path)

    #     unscaled_data_by_fold[fold_id] = MLP_Classifier(
    #         features_type=FEATURES_TYPE, config=dummy_model_unscaled
    #     ).load_data(train_path, test_path)for fold_id in tqdm(range(1, 6), desc="Loading data for each fold"):
    #     dummy_config = {
    #         "model_name": "mlp_classifier",
    #         "features_type": FEATURES_TYPE,
    #         "hidden_dims": [],
    #         "dropout_rate": 0.2,
    #         "learning_rate": 0.001,
    #         "batch_size": 256,
    #         "epochs": 10,
    #         "patience": 5,
    #         "decision_threshold": 0.5,
    #         "fold_id": fold_id,
    #         "dataset_version": DATASET_VERSION,
    #         "base_experiment_dir": "/tmp/financial_model_tuning",
    #         "oversample": False,
    #     }

    #     dummy_model_unscaled = {**dummy_config, "standardize": False}
    #     dummy_model_scaled = {**dummy_config, "standardize": True}

    #     train_path, test_path = load_cross_validation_path(dummy_config)

    #     scaled_data_by_fold[fold_id] = MLP_Classifier(
    #         features_type=FEATURES_TYPE, config=dummy_model_scaled
    #     ).load_data(train_path, test_path)

    #     unscaled_data_by_fold[fold_id] = MLP_Classifier(
    #         features_type=FEATURES_TYPE, config=dummy_model_unscaled
    #     ).load_data(train_path, test_path)

    # Load the data with Parallel processing
    scaled_data_by_folds = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(load_data_for_fold)(fold_id, True)
        for fold_id in tqdm(range(1, 6), desc="Loading scaled data for each fold")
    )
    scaled_data_by_fold = {
        fold_id: data for fold_id, data in zip(range(1, 6), scaled_data_by_folds)
    }

    unscaled_data_by_folds = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(load_data_for_fold)(fold_id, False)
        for fold_id in tqdm(range(1, 6), desc="Loading unscaled data for each fold")
    )
    unscaled_data_by_fold = {
        fold_id: data for fold_id, data in zip(range(1, 6), unscaled_data_by_folds)
    }

    # Now Runs the hyperparameter tuning
    trials = Trials()
    best = fmin(
        fn=lambda params: objective(
            params, model_type, scaled_data_by_fold, unscaled_data_by_fold
        ),
        space=params_space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        verbose=True,
        rstate=np.random.default_rng(SEED_TRAINING),  # For reproducibility
    )

    # Find the best trial based on the minimized loss (negative AUC)
    best_trial = sorted(
        trials.trials,
        key=lambda x: x["result"]["loss"] if "loss" in x["result"] else float("inf"),
    )[0]
    raw_best_params = best_trial["result"].get("params", {})
    best = raw_best_params
    best["base_experiment_dir"] = str(best["base_experiment_dir"])
    best = numpy_to_scalar(best)  # Convert numpy types to native Python types

    best_auc = -best_trial["result"]["loss"]
    logging.info(f"Best AUC: {best_auc:.4f} with parameters")
    best["val_auc"] = best_auc

    logging.info(f"Best parameters found: {best}")
    print(f"Best parameters found: {best}")
    # Save the best paramters to the tunning directory
    best_params_path = TUNING_DIR / f"best_params_{model_type.value}.json"
    with open(best_params_path, "w") as f:
        json.dump(numpy_to_scalar(best), f, indent=4)

    # Finally retrain an save the best model with the best parameters

    fold_aucs = []
    for fold_id in tqdm(
        range(1, 6), desc="Retraining best model and generating final metrics"
    ):
        config = {
            "fold_id": fold_id,
            "dataset_version": DATASET_VERSION,
            "standardize": best["standardize"],
            "decision_threshold": 0.5,
            "features_type": FEATURES_TYPE,
        }

        config.update(best)

        model_class = MODELS_CLASS_DICT[model_type]
        model = model_class(features_type=FEATURES_TYPE, config=config)

        # Use already loaded data
        if config["standardize"]:
            data = scaled_data_by_fold[fold_id]
        else:
            data = unscaled_data_by_fold[fold_id]

        X_train, y_train, X_val, y_val, X_test, y_test = data
        model.fit(X_train, y_train, X_val, y_val)
        test_metrics = model.evaluate(X_test, y_test, subset="test")
        logging.info(f"Test Metrics for fold {fold_id}: {test_metrics}")
        with open(TUNING_DIR / f"test_metrics_fold_{fold_id}.json", "w") as f:
            test_metrics = numpy_to_scalar(test_metrics)
            json.dump(test_metrics, f, indent=4)

        model.make_predictions_csv(
            X_test, y_test, TUNING_DIR / f"predictions_fold_{fold_id}.csv"
        )
        fold_aucs.append(test_metrics["auc_score"])

    # Create a file with the average AUC across all folds (include standard deviation)
    avg_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    with open(TUNING_DIR / "average_test_auc.json", "w") as f:
        avg_auc_data = {
            "average_auc": avg_auc,
            "std_auc": std_auc,
            "fold_aucs": fold_aucs,
        }
        json.dump(avg_auc_data, f, indent=4)


def main():
    """
    Main function to run the hyperparameter tuning script.
    """
    logging.info("Starting hyperparameter tuning...")
    trials = run_hyperparams_tuning()


if __name__ == "__main__":
    try:
        # Set start method for multiprocessing to 'spawn' for compatibility [1]
        multiprocessing.set_start_method("spawn", force=True)
        logging.info("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        logging.warning("Multiprocessing context already set. Could not force 'spawn'.")

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.info(
        f"Starting hyperparameter tuning for {MODEL_CLASS} with features type {FEATURES_TYPE} on dataset version {DATASET_VERSION}"
    )

    # Ensure the tuning directory exists
    TUNING_DIR.mkdir(parents=True, exist_ok=True)

    main()
