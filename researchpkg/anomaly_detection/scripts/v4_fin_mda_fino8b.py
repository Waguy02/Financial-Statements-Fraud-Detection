import os

from researchpkg.anomaly_detection.models.llm_unsloth.llm_classifier_softmax_financial_and_mda import (
    train_and_evaluate_financial_and_mda_softmax_model,
)

if __name__ == "__main__":
    lora_r = 8
    lora_alpha = 8
    num_layers_to_finetune = 32
    num_model_layers = 32
    lora_dropout = 0.05
    offline = int(os.environ.get("OFFLINE", 0))
    if offline:
        print("Running in offline mode. Ensure all necessary files are cached locally.")
        model_url = os.path.join(os.environ["HF_CACHE"], "Fino1-8B")
    else:
        model_url = "TheFinAI/Fino1-8B"
    model_alias = f'{model_url.split("/")[-1]}_Last_{num_layers_to_finetune}_Layers_lora_{lora_r}_{lora_alpha}_bs_8_dp{lora_dropout}_fixed_head_lr_1e-4'
    CONFIG = {
        "fold_id": int(os.environ["FOLD_ID"]),
        "model_url": model_url,
        "model_alias": model_alias,
        "max_context": 8500,
        "max_new_tokens": 1,
        "only_completion": True,
        "undersample": True,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 10,
        "learning_rate": 1e-4,
        "save_steps": 10,
        "run_eval_on_start": False,
        "lxt_explain": int(os.environ.get("LXT_EXPLAIN", 0)),
        "auto_continue": int(os.environ.get("AUTO_CONTINUE", 0)),
        "dataset_version": "company_isolated_splitting",
        "num_layers_to_finetune": num_layers_to_finetune,
        "num_model_layers": num_model_layers,
        "lora_target_modules": [
            "q_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
        "use_full_summary": True,
        "checkpoint_timestamp": str(os.environ.get("CHECKPOINT_TIMESTAMP"))
        if os.environ.get("CHECKPOINT_TIMESTAMP")
        else None,
    }

    train_and_evaluate_financial_and_mda_softmax_model(CONFIG)
