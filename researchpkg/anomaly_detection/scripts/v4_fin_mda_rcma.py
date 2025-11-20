import logging
import os
from pathlib import Path

from researchpkg.anomaly_detection.config import RCMA_EXPERIMENTS_DIR
from researchpkg.anomaly_detection.models.rcma.rcma_classifier import (
    train_and_evaluate_rcma_model,
)
from researchpkg.anomaly_detection.preprocessing.extended.sec_financial_preprocessing_quarterly_extended import (
    EXTENDED_FINANCIAL_FEATURES,
)

if __name__ == "__main__":

    rcma_config = {
        "sbert_model_name": "jinaai/jina-embeddings-v2-small-en",  # "all-MiniLM-L6-v2",
        "sbert_output_dim": 512,  # 384 for MiniLM
        "trainable_sbert_layers": 1,  # Set to >0 to fine-tune SBERT layers
        "num_original_financial_features": len(
            EXTENDED_FINANCIAL_FEATURES
        ),  # Auto-detected if possible
        "num_financial_groups": 7,
        "financial_embedding_dim": 512,
        "text_embedding_dim": 512,
        "mlp_hidden_dims_paper": [128],
        "dropout_rate": 0.05,
        "learning_rate": 1e-4,  # Might need adjustment for trainable SBERT
        "batch_size": 8,
        "val_batch_size": 8,
        "epochs": 10,  # Increased for actual run
        "patience": 7,
        # "decision_threshold": 0.5, # Removed
        "pos_weight_beta": 0.75,  # For FocalLoss
        "focal_gamma": 2.0,  # For FocalLoss
        # "consistency_loss_weight": 0.05,
        "consistency_loss_weight": 0.2,
        "oversample": False,  # Set to True to oversample fraud cases in training
        "dataset_version": "company_isolated_splitting",
        "fold_id": int(os.environ["FOLD_ID"]),
        "target_modules": ["query","value"],  # For LoRA fine-tuning
        "max_mda_length": 8192,
        # "embedding_prefix_for_sbert": "classification:",
        "gradient_accumulation_steps": 4,
    }
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Example of running:
    train_and_evaluate_rcma_model(rcma_config)
    print("RCMA Model training and evaluation example finished.")
