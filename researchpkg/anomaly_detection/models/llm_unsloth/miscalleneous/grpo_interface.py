import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from tqdm import tqdm
from trl import GRPOConfig, GRPOTrainer
from researchpkg.anomaly_detection.models.utils import (
    correctness_reward_func,
    extract_xml_answer,
    get_last_checkpoint,
    llm_generate,
    llm_vllm_generate,
    llm_fast_generate,
    xmlcount_reward_func
)
from researchpkg.anomaly_detection.models.llm_unsloth.miscalleneous.base_llm_classifier import (
    EvaluationCallback,
    PermutableUndersamplingDataset
)
from tqdm import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
import json
import logging
import os
import random
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import Dataset
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


class GRPO_EvaluationCallback(EvaluationCallback):
    """Custom callback for GRPO trainer evaluation at the end of each epoch"""

    def __init__(
        self,
        trainer,
        tokenizer,
        log_dir,
        max_length,
        max_new_tokens,
        run_eval_on_start=False,
    ):
        super().__init__(trainer, 
                        tokenizer, 
                        log_dir, 
                        max_length, 
                        max_new_tokens,
                        run_eval_on_start)
    
    def extract_answer_prediction(self, prediction_text):
        return extract_xml_answer(prediction_text)
    
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        pass
        

    def on_train_begin(self, args, state, control, **kwargs):
        if state.epoch == 0 and self.run_eval_on_start:
            logging.info("Running evaluation at the beginning of training...")
            self.on_epoch_end(args, state, control, **kwargs)
        

    def on_epoch_end(self, args, state, control, **kwargs):
        """Run evaluation at the end of each epoch."""
        with torch.no_grad():
            model  = self.trainer.model
            vllm_client = None
            try:
                vllm_client = self.trainer.vllm_client
                # vllm_client.update_model_params(model) # No ncessary here
            except Exception as e:
                logging.error(
                    f"No vllm client found for GRPO: {e}. Using native inference"
                )
                # If no vllm client, load lora weights
            if vllm_client is None:
                FastLanguageModel.for_inference(model)    
                
            # Extract true labels and get predictions
            true_labels = []
            predicted_labels = []
            ciks = []
            sics = []
            quarters = []
            glabels = []

            # Process a single example function for parallel processing
            def process_batch(examples, model, tokenizer, vllm_client=None):
                batch_results = []
                
                # Prepare batch inputs
                batch_prompts = []
                
                for example in examples:
                    # Extract data from example
                    prompt = example["prompt"]
                    
                    # Generate prediction
                    chat_prompt = prompt + self.tokenizer.completion_instruction
                    
                    batch_prompts.append(chat_prompt)
                
                # Generate predictions for the batch
                if vllm_client:
                    predictions = llm_vllm_generate(
                        model,
                        tokenizer,
                        vllm_client,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                    )
                elif self.fast_generation:
                    predictions = llm_fast_generate(
                        model,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                        self.current_lora_request,
                    )
                else:
                    predictions = llm_generate(
                        model,
                        tokenizer,
                        batch_prompts,
                        self.max_length,
                        self.max_new_tokens,
                    )
                
                # Process each prediction
                for i, prediction_text in enumerate(predictions):
                    example = examples[i]
                    answer = example["answer"]
                    
                    # Extract the answer from the XML format
                    extracted_answer = extract_xml_answer(prediction_text.split(tokenizer.completion_instruction)[-1].strip())
                    
                    # Clean up prediction to get just the label
                    if extracted_answer not in ["Fraud", "Not Fraud"]:
                        if ("fraud" in extracted_answer.lower() and "not fraud" not in extracted_answer.lower()):
                            extracted_answer = "Fraud"
                        else:
                            extracted_answer = "Not Fraud"
                    
                    batch_results.append({
                        "true_label": answer,
                        "predicted_label": extracted_answer,
                        "prediction": prediction_text,
                        "cik": int(example["cik"]),
                        "sic": example["sic"],
                        "quarter": example["quarter"],
                        "glabels": example["glabels"]
                    })
                    
                return batch_results

            with torch.no_grad():
                vllm_client = None
                try:
                    vllm_client = self.trainer.vllm_client
                except Exception as e:
                    logging.error(f"No vllm client found in GRPO callback: {e}")
                
                # Process in batches
                batch_size = self.trainer.args.per_device_eval_batch_size
                dataset_length = len(self.trainer.eval_dataset)
                
                for i in tqdm(range(0, dataset_length, batch_size), 
                            desc=f"GRPO Evaluation at epoch {int(state.epoch)} (batch size: {batch_size})"):
                    # Get batch of examples
                    batch_examples = [self.trainer.eval_dataset[j] for j in range(i, min(i + batch_size, dataset_length))]
                    
                    # Process the batch
                    batch_results = process_batch(batch_examples, self.trainer.model, self.tokenizer, vllm_client)
                    
                    # Collect results
                    for result,batch_example in zip(batch_results, batch_examples):
                        true_labels.append(result["true_label"])
                        predicted_labels.append(result["predicted_label"])
                        ciks.append(result["cik"])
                        sics.append(result["sic"])
                        quarters.append(result["quarter"])
                        glabels.append(result["glabels"])
                        
                        # Log sample predictions occasionally
                        if state.epoch == 0 or random.random() < 0.001:
                            logging.info(f"Prompt: {batch_example['text']}")
                            logging.info(f"Prediction: {result['prediction']}")
                            logging.info(f"Answer: {batch_example['answer']}")
                            logging.info(f"True: {result['true_label']}, Pred: {result['predicted_label']}")
                            logging.info(f"CIK: {result['cik']}, SIC: {result['sic']}, Quarter: {result['quarter']}")

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, pos_label="Fraud")
            recall = recall_score(true_labels, predicted_labels, pos_label="Fraud")
            f1 = f1_score(true_labels, predicted_labels, pos_label="Fraud")
            macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
            weighted_f1 = f1_score(true_labels, predicted_labels, average="weighted")
            report = classification_report(true_labels, predicted_labels)

            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"Precision: {precision:.4f}")
            logging.info(f"Recall: {recall:.4f}")
            logging.info(f"F1 Score: {f1:.4f}")
            logging.info(f"Macro F1 Score: {macro_f1:.4f}")
            logging.info(f"Weighted F1 Score: {weighted_f1:.4f}")
            logging.info(f"Classification Report:\n{report}")

            # Save metrics to a file
            with open(os.path.join(self.log_dir, f"metrics_epoch_{int(state.epoch)}.json"), "w") as f:
                json.dump(
                    {   
                        "epoch": int(state.epoch),
                        "accuracy": float(accuracy),
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "macro_f1": float(macro_f1),
                        "weighted_f1": float(weighted_f1),
                        "report": report,
                    },
                    f,
                    indent=2,
                )

            # Save detailed test predictions to CSV
            val_results_df = pd.DataFrame(
                {
                    "cik": ciks,
                    "sic": sics,
                    "quarter": quarters,
                    "y_true": true_labels,
                    "y_pred": predicted_labels,
                    "glabels": glabels,
                }
            )

            val_csv_path = os.path.join(self.log_dir, f"validation_predictions_epoch_{int(state.epoch)}.csv")
            val_results_df.to_csv(val_csv_path, index=False)
            logging.info(f"Saved detailed test predictions to {val_csv_path}")

            # Keep model in train mode for further training
            model.train()
            
            if vllm_client:
                FastLanguageModel.for_training(model)
            return control

class GRPOMixin:
    """
    Mixin providing shared GRPO llm_tune and evaluate methods.
    Assumes host class defines:
      - self.model, self.tokenizer, self.config, self.log_dir, self.checkpoint_timestamp
      - self.per_device_train_batch_size, self.per_device_eval_batch_size
      - self.gradient_accumulation_steps, self.max_length
      - self.model_url, self.model_alias
      - self.generate_prompt(...)
      - save_model(...)
    """
    def llm_tune(self, train_df, val_df, num_epochs=1, learning_rate=1e-5, save_steps=100):
        logging.info("Starting GRPO tuning...")
        # 1) dump experiment_config
        config_run = {
            "model_url": self.model_url,
            "model_alias": self.model_alias,
            "max_length": self.max_length,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "save_steps": save_steps,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "base_config": self.config,
        }
        with open(self.log_dir / "experiment_config.yaml", "w") as f:
            yaml.dump(config_run, f, indent=2)

        # 2) build datasets from prompts
        train_list = []
        for _, row in tqdm(train_df.iterrows(), desc="Preparing training data"):
            p = self.generate_prompt(row)
            if p is not None:
                train_list.append(p)
        val_list = []
        for idx, (_, row) in enumerate(tqdm(val_df.iterrows(), desc="Preparing validation data")):
            p = self.generate_prompt(row, idx)
            if p is not None:
                val_list.append(p)
        train_dataset = Dataset.from_list(train_list)
        val_dataset = Dataset.from_list(val_list)

        # 3) configure GRPO
        grpo_args = GRPOConfig(
            adam_beta1=self.config.get("adam_beta1", 0.9),
            adam_beta2=self.config.get("adam_beta2", 0.99),
            bf16=self.config.get("bf16", True),
            dataloader_num_workers=min(1, os.cpu_count()),
            dataloader_persistent_workers=False,
            do_eval=self.config.get("do_eval", False),
            eval_strategy=self.config.get("eval_strategy", "no"),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=learning_rate,
            local_rank=self.lora_r,
            logging_dir=str(self.log_dir),
            logging_steps=self.config.get("logging_steps", 2),
            lr_scheduler_type=self.config.get("lr_scheduler_type", "cosine"),
            max_completion_length=self.config["max_new_tokens"],
            num_generations=self.config["num_generations"],
            num_train_epochs=num_epochs,
            optim=self.config.get("optim", "paged_adamw_8bit"),
            output_dir=str(self.log_dir),
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            per_device_train_batch_size=self.per_device_train_batch_size,
            report_to=self.config.get("report_to", "tensorboard"),
            save_strategy=self.config.get("save_strategy", "steps"),
            save_steps=self.config.get(
                "save_steps",
                int((len(train_dataset) // self.per_device_train_batch_size) * 0.1),
            ),
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            weight_decay=self.config.get("weight_decay", 0.1),
            use_vllm=self.config.get("use_vllm", True),
            gradient_checkpointing=self.config.get("gradient_checkpointing", True),
        )
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[correctness_reward_func,
                        #   xmlcount_reward_func,
                          ],
            
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=grpo_args,
        )

        # 4) optional undersampling
        if self.config.get("undersample", False):
            logging.info("Using dynamic undersampling for training...")
            labels = np.array(train_df["is_fraud"].tolist())
            trainer.undersample = True
            trainer.train_dataset = PermutableUndersamplingDataset(
                dataset=trainer.train_dataset,
                train_labels=labels,
                max_length=self.max_length,
                tokenizer=self.tokenizer,
            )

        # 5) add callback
        
        callback = GRPO_EvaluationCallback(
            trainer=trainer,
            tokenizer=self.tokenizer,
            log_dir=self.log_dir,
            max_length=self.max_length,
            max_new_tokens=self.config["max_new_tokens"],
            run_eval_on_start=self.config.get("run_eval_on_start", True),
        )
        trainer.add_callback(callback)

        # 6) train/resume
        if self.checkpoint_timestamp:
            trainer.train(resume_from_checkpoint=get_last_checkpoint(self.log_dir))
        else:
            trainer.train()
        logging.info("GRPO tuning complete.")

        # 7) save model
        self.save_model(filepath=self.log_dir)
        return self

    def evaluate(self, test_df, vllm_client=None):
        logging.info("Starting evaluation...")
        self.model.eval()
        # build test dataset
        data_list = []
        for _, row in tqdm(test_df.iterrows(), desc="Preparing test data"):
            p = self.generate_prompt(row)
            if p is not None:
                data_list.append(p)
        test_dataset = Dataset.from_list(data_list)

        true_labels, pred_labels, ciks, sics, quarters, glabels = [], [], [], [], [], []
        from researchpkg.anomaly_detection.models.utils import llm_vllm_generate, llm_generate, extract_xml_answer, llm_fast_generate

        with torch.no_grad():
            for i in tqdm(range(0, len(test_dataset), self.per_device_eval_batch_size),
                          desc=f"Evaluating on test set (batch size: {self.per_device_eval_batch_size})"):
                batch = test_dataset[i : i + self.per_device_eval_batch_size]
                prompts = [
                    msg
                    for ex in batch
                    for msg in ex["prompt"]
                ]
                if vllm_client:
                    outputs = llm_vllm_generate(
                        self.model,
                        self.tokenizer,
                        vllm_client,
                        prompts,
                        self.max_length,
                        self.config["max_new_tokens"],
                    )
                elif hasattr(self.trainer, "fast_generation") and self.trainer.fast_generation:
                    outputs = llm_fast_generate(
                        self.model,
                        prompts,
                        self.max_length,
                        self.config["max_new_tokens"],
                        self.current_lora_request,
                    )
                else:
                    outputs = llm_generate(
                        self.model,
                        self.tokenizer,
                        prompts,
                        self.max_length,
                        self.config["max_new_tokens"],
                    )
                for ex, out in zip(batch, outputs):
                    ans = extract_xml_answer(
                        out.split(self.tokenizer.completion_instruction)[-1].strip()
                    )
                    if ans not in ["Fraud", "Not Fraud"]:
                        ans = "Fraud" if "fraud" in ans.lower() and "not fraud" not in ans.lower() else "Not Fraud"
                    true_labels.append(ex["answer"])
                    pred_labels.append(ans)
                    ciks.append(int(ex["cik"]))
                    sics.append(ex["sic"])
                    quarters.append(ex["quarter"])
                    glabels.append(ex["glabels"])

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, pos_label="Fraud")
        recall = recall_score(true_labels, pred_labels, pos_label="Fraud")
        f1 = f1_score(true_labels, pred_labels, pos_label="Fraud")
        report = classification_report(true_labels, pred_labels)
        logging.info(f"Test accuracy: {accuracy:.4f}")

        # dump metrics & csv
        with open(os.path.join(self.log_dir, "test_metrics.json"), "w") as f:
            json.dump(
                {"accuracy": float(accuracy), "precision": float(precision),
                 "recall": float(recall), "f1": float(f1), "report": report},
                f,
                indent=2,
            )
        pd.DataFrame(
            {
                "cik": ciks,
                "sic": sics,
                "quarter": quarters,
                "y_true": true_labels,
                "y_pred": pred_labels,
                "glabels": glabels,
            }
        ).to_csv(os.path.join(self.log_dir, "test_predictions.csv"), index=False)

        return accuracy, report
