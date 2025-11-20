"""
Base Gemini classifier for fraud detection that returns probabilities by analyzing logprobs.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd
import yaml
from google import genai
from google.genai import types
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from researchpkg.anomaly_detection.config import (
    EXPERIMENTS_DIR,
    VERTEX_JSON_KEYS_PATH,
    SEED_TRAINING,
)
from researchpkg.anomaly_detection.models.utils import (
    drop_random_keys,
    get_train_test_splitter,
    load_cross_validation_path,
)
from researchpkg.utils import configure_logger
from google.oauth2 import service_account
# Constants for classification labels
NOT_FRAUD_TOKEN = "NO"
FRAUD_TOKEN = "YES"
NOT_FRAUD_LABEL_ID = 0
FRAUD_LABEL_ID = 1

# Standard completion instruction used across models
COMPLETION_INSTRUCTION = "\n## My answer is: \n"

# Maximum requests per minute for Gemini API
MAX_RPM_GEMINI = 15

# Maximum number of retries for API calls
MAX_RETRY = 3

# Default model to use
DEFAULT_MODEL = "gemini-2.0-flash"

class GeminiClassifier:
    """
    Base class for fraud classification using Gemini API with probability outputs.
    """
    
    def __init__(self, config):
        """
        Initialize the Gemini classifier.
        
        Args:
            config (dict): Configuration dictionary with model settings
        """
        self.config = config
        self.model_name = config.get("model_name", DEFAULT_MODEL)
        
        # Set up experiment directory
        self.experiments_dir = config.get(
            "experiments_dir", EXPERIMENTS_DIR / f"llm_gemini_fraud_classifier"
        )
        if not isinstance(self.experiments_dir, Path):
            self.experiments_dir = Path(self.experiments_dir)
            
        # Create timestamp and log directory
        fold_id = self.config.get("fold_id", 0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = (
            self.experiments_dir
            / f"{self.model_name.replace('-', '_')}"
            / f"fold_{fold_id}"
            / f"{timestamp}"
        )
        
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
        # Set up logging
        self.log_file = self.log_dir / "experiment.log"
        configure_logger(self.log_file, logging.INFO)
        
        logging.info(f"Initialized Gemini classifier using model: {self.model_name}")
        logging.info(f"Log directory: {self.log_dir}")
        
        # Load API keys and set up round-robin rotation
        self.api_keys = self._load_api_keys()
        self.api_key_cycle = cycle(self.api_keys)
        self.current_api_key = next(self.api_key_cycle)
        
        # Track request timestamps for rate limiting
        self.request_timestamps = []
        
        # Save experiment config
        self._save_experiment_config()
    
    def _load_api_keys(self) -> List[str]:
        """
        Load Gemini API keys from the JSON file with round-robin rotation.
        
        Returns:
            List[str]: List of valid API keys
        """
        if not VERTEX_JSON_KEYS_PATH.exists():
            raise FileNotFoundError(
                f"Gemini API Keys file not found at {VERTEX_JSON_KEYS_PATH}. "
                "Create this file with API keys to use this classifier."
            )
            
        with open(VERTEX_JSON_KEYS_PATH, "r") as f:
            data = json.load(f)
            
        # Extract valid keys (not disabled)
        api_keys = []
        for entry in data:
            if not entry.get("disabled", False):
                key = entry["key"]
                service_account= entry["service_account"]
                project_id = entry["project_id"]
                location = entry["location"]
                email = entry.get("email", "no email provided")
                api_keys.append((key, service_account, project_id, location))
                logging.info(f"Loaded API key for email: {email}")
                
        if not api_keys:
            raise ValueError("No valid API keys found in the keys file.")
            
        logging.info(f"Loaded {len(api_keys)} API keys for round-robin rotation")
        return api_keys
    
    def _save_experiment_config(self):
        """
        Save the experiment configuration to a YAML file.
        """
        config_to_save = {k: v for k, v in self.config.items()}
        config_to_save["model_name"] = self.model_name
        config_to_save["log_dir"] = str(self.log_dir)
        config_to_save["timestamp"] = self.log_dir.name
        
        config_path = self.log_dir / "experiment_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_to_save, f, default_flow_style=False)
            
        logging.info(f"Saved experiment configuration to {config_path}")
    
    def _get_next_api_key(self) -> str:
        """
        Get the next API key in rotation, respecting rate limits.
        
        Returns:
            str: API key to use for the next request
        """
        # Clean up old timestamps (older than 60 seconds)
        current_time = time.time()
        self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        
        # Check if we're approaching rate limit
        if len(self.request_timestamps) >= MAX_RPM_GEMINI:
            # Calculate time to wait
            oldest_timestamp = min(self.request_timestamps)
            wait_time = 61 - (current_time - oldest_timestamp)
            if wait_time > 0:
                logging.info(f"Rate limit approaching. Waiting {wait_time:.2f} seconds.")
                time.sleep(wait_time)
                # Clean timestamps again after waiting
                current_time = time.time()
                self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
        
        # Rotate to next key
        self.current_api_key = next(self.api_key_cycle)
        self.request_timestamps.append(time.time())
        return self.current_api_key
    
    def generate_with_probabilities(
        self, prompt: str, temperature: float = 0.0
    ) -> Tuple[str, Dict[str, float]]:
        """
        Generate a response using Gemini with probability information.
        
        Args:
            prompt (str): The prompt text to send to the model
            temperature (float): Temperature for generation, use 0.0 for consistency
            
        Returns:
            Tuple[str, Dict[str, float]]: (prediction, token_probabilities)
        """
        _,service_account_info, project,location = self._get_next_api_key()
        credentials = service_account.Credentials.from_service_account_info(service_account_info,
                scopes=["https://www.googleapis.com/auth/cloud-platform"])
    
        
        gemini_client = genai.Client(
            credentials=credentials,
            project=project,
            location=location,
            vertexai=True,
        )
        
        
        try:
            # Generate with logprobs enabled
            response = gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=1,  # Only generate 1 token for classification
                    response_logprobs=True,  # Return probability scores
                    logprobs=2,  # Top 2 probabilities (YES/NO)
                    safety_settings=[
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                        types.SafetySetting(
                            category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                            threshold=types.HarmBlockThreshold.BLOCK_NONE,
                        ),
                    ],
                ),
            )
            
            # Extract the response text
            prediction = response.text.strip()
            
            # Extract probability information from logprobs
            token_probs = {}
            
            candidate = response.candidates[0]
            top_tokens = candidate.logprobs_result.top_candidates[0].candidates
            
            for token_info in top_tokens[0]:
                token = token_info.token.strip()
                if token in [FRAUD_TOKEN, NOT_FRAUD_TOKEN]:
                    token_probs[token] = token_info.log_probability
        
            
            if NOT_FRAUD_TOKEN not in token_probs:
                token_probs[NOT_FRAUD_TOKEN] = -100  # Very low probability
            if FRAUD_TOKEN not in token_probs:
                token_probs[FRAUD_TOKEN] = -100  # Very low probability
            
            # Convert log probabilities to actual probabilities
            logprobs = [token_probs[NOT_FRAUD_TOKEN], token_probs[FRAUD_TOKEN]]
            max_logprob = max(logprobs)
            exp_logprobs = [np.exp(logprob - max_logprob) for logprob in logprobs]
            sum_exp_logprobs = sum(exp_logprobs)
            probabilities = {
                NOT_FRAUD_TOKEN: exp_logprobs[0] / sum_exp_logprobs,
                FRAUD_TOKEN: exp_logprobs[1] / sum_exp_logprobs,
            }
          
            return prediction, probabilities
            
        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            # Return a default response with zero probabilities
            return "ERROR", {NOT_FRAUD_TOKEN: 0.5, FRAUD_TOKEN: 0.5}
    
    def find_best_threshold(
        self, y_true: List[int], y_scores: List[float], epoch: int, 
        min_threshold: float = 0.05, max_threshold: float = 0.95
    ) -> Tuple[float, float]:
        """
        Find the best classification threshold to maximize F1 score for the positive class (Fraud).
        
        Args:
            y_true: True binary labels (0 for Not Fraud, 1 for Fraud)
            y_scores: Predicted probabilities for the Fraud class
            epoch: Current epoch (for logging)
            min_threshold: Minimum threshold to consider
            max_threshold: Maximum threshold to consider
            
        Returns:
            Tuple[float, float]: (best_threshold, best_f1_score)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        
        # Filter thresholds to the specified range
        precs, recs, thrs = [], [], []
        for p, r, t in zip(precision, recall, thresholds):
            if min_threshold <= t <= max_threshold:
                precs.append(p)
                recs.append(r)
                thrs.append(t)
        
        if not thrs:
            logging.warning("No thresholds in the specified range. Using default 0.5 threshold.")
            return 0.5, f1_score(y_true, [1 if s >= 0.5 else 0 for s in y_scores], zero_division=0)
        
        precision, recall, thresholds = np.array(precs), np.array(recs), np.array(thrs)
        
        # Calculate F1 scores for each threshold
        f1_scores = []
        for i in range(len(thresholds)):
            p = precision[i]
            r = recall[i]
            if p + r == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * (p * r) / (p + r))
        
        # Save plot of F1 vs threshold
        f1_scores = np.array(f1_scores)
        thresholds = np.array(thresholds)
        
        fig = plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, marker='o', linestyle='-', color='b')
        plt.title(f"F1 Score vs. Threshold (Epoch {epoch})")
        plt.xlabel("Threshold")
        plt.ylabel("F1 Score")
        plt.grid()
        plt.savefig(os.path.join(self.log_dir, f"f1_vs_threshold_epoch_{epoch}.png"))
        
        # Find threshold with maximum F1 score
        best_f1_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_f1_idx]
        best_threshold = thresholds[best_f1_idx]
        
        # Clip threshold to avoid edge cases
        best_threshold = np.clip(best_threshold, 1e-6, 1 - 1e-6)
        logging.info(f"Optimized threshold: {best_threshold:.4f} -> F1 Score: {best_f1:.4f}")
        
        return best_threshold, best_f1
    
    def find_best_auc_threshold(
        self, y_true: List[int], y_scores: List[float], epoch: int, 
        min_threshold: float = 0.05, max_threshold: float = 0.95
    ) -> Tuple[float, float]:
        """
        Find the best classification threshold to maximize AUC score for the positive class (Fraud).
        
        Args:
            y_true: True binary labels (0 for Not Fraud, 1 for Fraud)
            y_scores: Predicted probabilities for the Fraud class
            epoch: Current epoch (for logging)
            min_threshold: Minimum threshold to consider
            max_threshold: Maximum threshold to consider
            
        Returns:
            Tuple[float, float]: (best_threshold, best_auc_score)
        """
        # Get unique thresholds from the predicted probabilities
        thresholds = np.unique(y_scores)
        thresholds = thresholds[(thresholds >= min_threshold) & (thresholds <= max_threshold)]
        
        if len(thresholds) == 0:
            return 0.5, roc_auc_score(y_true, y_scores)
        
        best_auc = 0.0
        best_threshold = 0.5  # Default
        
        for threshold in thresholds:
            y_pred = (np.array(y_scores) >= threshold).astype(int)
            try:
                current_auc = roc_auc_score(y_true, y_pred)
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_threshold = threshold
            except ValueError:
                continue
        
        # Also calculate F1 at this threshold for reference
        y_pred_at_best = (np.array(y_scores) >= best_threshold).astype(int)
        f1_at_best = f1_score(y_true, y_pred_at_best)
        
        logging.info(f"Epoch {epoch} - Best AUC: {best_auc:.4f} at threshold: {best_threshold:.4f} (F1: {f1_at_best:.4f})")
        
        return best_threshold, best_auc
    
    def evaluate(
        self, data_df: pd.DataFrame, fold: str = "val", epoch: int = 0, 
        threshold: Optional[float] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Evaluate the model on a dataset with threshold optimization.
        
        Args:
            data_df: DataFrame with data to evaluate
            fold: Name of the fold ("val" or "test")
            epoch: Current epoch (for logging)
            threshold: Optional threshold to use (if None, optimize on validation)
            
        Returns:
            Tuple[Dict, pd.DataFrame]: (metrics_dict, predictions_df)
        """
        logging.info(f"Starting evaluation on {fold} set (epoch {epoch})")
        
        if data_df is None or len(data_df) == 0:
            logging.warning(f"Empty {fold} dataset. Skipping evaluation.")
            return {}, pd.DataFrame()
        
        # Store predictions and ground truth
        true_labels = []
        predicted_probs = []
        predictions = []
        ciks = []
        sics = []
        quarters = []
        
        # Process each example
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc=f"Evaluating {fold} set"):
            retry_count = 0
            success = False
            
            while not success and retry_count < MAX_RETRY:
                try:
                    # Generate prompt for this example
                    prompt = self.generate_prompt(row, drop_rate=0.0)  # No dropout for eval
                    
                    # Get prediction and probabilities
                    prediction, probabilities = self.generate_with_probabilities(prompt)
                    
                    # Extract fraud probability
                    fraud_prob = probabilities.get(FRAUD_TOKEN, 0.0)
                    
                    # Store results
                    true_label = 1 if row["is_fraud"] else 0
                    true_labels.append(true_label)
                    predicted_probs.append(fraud_prob)
                    predictions.append(prediction)
                    ciks.append(int(row["cik"]))
                    sics.append(str(row["sic"]).zfill(4))
                    quarters.append(f"{row['year']}{row['quarter']}")
                    
                    # Log occasional examples
                    if idx % 50 == 0:
                        logging.info(f"Example {idx}: true={true_label}, "
                                    f"pred={prediction}, fraud_prob={fraud_prob:.4f}")
                    
                    success = True
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count < MAX_RETRY:
                        wait_time = 2 ** retry_count  # Exponential backoff
                        logging.warning(f"Error on example {idx}, retry {retry_count}/{MAX_RETRY} after {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logging.error(f"Failed after {MAX_RETRY} attempts on example {idx}: {str(e)}")
                        # Add a default prediction on failure
                        true_label = 1 if row["is_fraud"] else 0
                        true_labels.append(true_label)
                        predicted_probs.append(0.5)  # Default to 0.5 probability on failure
                        predictions.append("ERROR")
                        ciks.append(int(row["cik"]))
                        sics.append(str(row["sic"]).zfill(4))
                        quarters.append(f"{row['year']}{row['quarter']}")
        
        # Find best threshold on validation or use provided threshold on test
        if threshold is None and fold == "val":
            threshold, best_f1 = self.find_best_threshold(true_labels, predicted_probs, epoch)
        elif threshold is None:
            threshold = 0.5  # Default if not optimizing
        
        # Apply threshold to get binary predictions
        thresholded_preds = [1 if prob >= threshold else 0 for prob in predicted_probs]
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, thresholded_preds)
        precision = precision_score(true_labels, thresholded_preds, zero_division=0)
        recall = recall_score(true_labels, thresholded_preds, zero_division=0)
        f1 = f1_score(true_labels, thresholded_preds, zero_division=0)
        
        # Additional metrics
        macro_f1 = f1_score(true_labels, thresholded_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(true_labels, thresholded_preds, average="weighted", zero_division=0)
        auc_score = roc_auc_score(true_labels, predicted_probs) if len(set(true_labels)) > 1 else 0.5
        
        # Generate classification report
        token_true_labels = [FRAUD_TOKEN if label == 1 else NOT_FRAUD_TOKEN for label in true_labels]
        token_pred_labels = [FRAUD_TOKEN if pred == 1 else NOT_FRAUD_TOKEN for pred in thresholded_preds]
        report = classification_report(
            token_true_labels, 
            token_pred_labels,
            labels=[FRAUD_TOKEN, NOT_FRAUD_TOKEN],
            target_names=["Fraud", "Not Fraud"],
            zero_division=0
        )
        
        # Log metrics
        logging.info(f"--- {fold.upper()} Metrics (Threshold: {threshold:.4f}) ---")
        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision (Fraud): {precision:.4f}")
        logging.info(f"Recall (Fraud): {recall:.4f}")
        logging.info(f"F1 Score (Fraud): {f1:.4f}")
        logging.info(f"Macro F1: {macro_f1:.4f}")
        logging.info(f"Weighted F1: {weighted_f1:.4f}")
        logging.info(f"AUC: {auc_score:.4f}")
        logging.info(f"Classification Report:\n{report}")
        
        # Create metrics dictionary
        metrics = {
            "accuracy": float(accuracy),
            "precision_fraud": float(precision),
            "recall_fraud": float(recall),
            "f1_fraud": float(f1),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "auc_score": float(auc_score),
            "threshold": float(threshold),
            "report": report,
            "num_samples": len(true_labels),
            "num_fraud_samples": sum(true_labels),
        }
        
        # Save metrics to file
        metrics_path = self.log_dir / f"{fold}_metrics_epoch_{epoch}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Create predictions DataFrame
        preds_df = pd.DataFrame({
            "cik": ciks,
            "sic": sics,
            "quarter": quarters,
            "y_true": true_labels,
            "y_pred": thresholded_preds,
            "fraud_probability": predicted_probs,
            "raw_prediction": predictions,
        })
        
        # Save predictions to CSV
        preds_path = self.log_dir / f"{fold}_predictions_epoch_{epoch}.csv"
        preds_df.to_csv(preds_path, index=False)
        
        return metrics, preds_df
    
    def train_and_evaluate(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
        epochs: int = 1
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Run the full training and evaluation workflow.
        Note: Gemini doesn't have actual training, but we structure it like the other models.
        
        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            epochs: Number of "epochs" to run (mostly for consistency with other models)
            
        Returns:
            Tuple[Dict, pd.DataFrame]: (test_metrics, test_predictions_df)
        """
        logging.info(f"Starting Gemini classifier workflow with {epochs} evaluation rounds")
        
        # For each "epoch" (round of evaluation)
        best_val_f1 = -1
        best_threshold = 0.5
        best_epoch = 0
        
        for epoch in range(epochs):
            logging.info(f"Starting epoch {epoch+1}/{epochs}")
            
            # Evaluate on validation set
            val_metrics, val_preds = self.evaluate(val_df, fold="val", epoch=epoch)
            
            # Track best validation performance and threshold
            if val_metrics and val_metrics["f1_fraud"] > best_val_f1:
                best_val_f1 = val_metrics["f1_fraud"]
                best_threshold = val_metrics["threshold"]
                best_epoch = epoch
                logging.info(f"New best threshold: {best_threshold:.4f} (F1: {best_val_f1:.4f})")
        
        # Final evaluation on test set using best threshold
        logging.info(f"Evaluating test set with best threshold ({best_threshold:.4f}) from epoch {best_epoch}")
        test_metrics, test_preds = self.evaluate(
            test_df, fold="test", epoch=best_epoch, threshold=best_threshold
        )
        
        # Save final results
        with open(self.log_dir / "final_metrics.json", "w") as f:
            json.dump({
                "best_epoch": best_epoch,
                "best_val_f1": float(best_val_f1),
                "best_threshold": float(best_threshold),
                "test_metrics": test_metrics
            }, f, indent=2)
        
        return test_metrics, test_preds
    
    def load_data(self, train_path=None, test_path=None):
        """
        Load and process train and test datasets.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        logging.info(f"Loading train data from {train_path}")
        train_df = pd.read_csv(train_path)
        
        logging.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        
        # Process the dataframes
        train_df = self._process_loaded_data(train_df)
        test_df = self._process_loaded_data(test_df)
        
        # Split train into train and validation
        val_split_size = self.config.get("validation_split_size", 0.1)
        splitter = get_train_test_splitter(self.config)
        train_df, val_df = splitter(train_df, test_size=val_split_size)
        
        # Handle oversampling if specified
        if self.config.get("oversample", False):
            train_df = self._oversample_fraud_cases(train_df)
        
        logging.info(f"Train data size: {len(train_df)} (Fraud: {train_df['is_fraud'].sum()})")
        logging.info(f"Validation data size: {len(val_df)} (Fraud: {val_df['is_fraud'].sum()})")
        logging.info(f"Test data size: {len(test_df)} (Fraud: {test_df['is_fraud'].sum()})")
        
        return train_df, val_df, test_df
    
    def _process_loaded_data(self, df):
        """
        Process loaded data - to be implemented by subclasses.
        
        Args:
            df: DataFrame to process
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Ensure is_fraud is consistent numeric format
        if 'is_fraud' in df.columns:
            df['is_fraud'] = df['is_fraud'].astype(int)
        
        return df
    
    def _oversample_fraud_cases(self, df):
        """
        Oversample fraud cases to balance the dataset.
        
        Args:
            df: DataFrame to oversample
            
        Returns:
            pd.DataFrame: Oversampled DataFrame
        """
        fraud_df = df[df["is_fraud"] == 1]
        non_fraud_df = df[df["is_fraud"] == 0]
        
        num_fraud = len(fraud_df)
        num_non_fraud = len(non_fraud_df)
        
        if num_fraud >= num_non_fraud:
            logging.info(f"No oversampling needed: Fraud cases ({num_fraud}) >= Non-fraud cases ({num_non_fraud}).")
            return df
        
        # Oversample fraud cases
        oversampled_fraud_df = fraud_df.sample(
            n=num_non_fraud, replace=True, random_state=SEED_TRAINING
        )
        
        # Combine and shuffle
        oversampled_df = pd.concat([oversampled_fraud_df, non_fraud_df], ignore_index=True)
        oversampled_df = oversampled_df.sample(frac=1, random_state=SEED_TRAINING).reset_index(drop=True)
        
        logging.info(f"Oversampling complete: {len(oversampled_df)} total samples")
        
        return oversampled_df
    
    def generate_prompt(self, row, drop_rate=0.0):
        """
        Generate a prompt for the given row - to be implemented by subclasses.
        
        Args:
            row: DataFrame row
            drop_rate: Rate to drop features for data augmentation
            
        Returns:
            str: Formatted prompt
        """
        raise NotImplementedError("Subclasses must implement generate_prompt")
    
    def format_financials(self, financials, drop_rate=0.0):
        """
        Format financial data as a string.
        
        Args:
            financials: Dictionary of financial features
            drop_rate: Rate to drop features for data augmentation
            
        Returns:
            str: Formatted string of financial data
        """
        # To be implemented by subclasses as needed
        raise NotImplementedError("Subclasses must implement format_financials if using financial data")

def train_and_evaluate_gemini_model(model_class, config=None, train_path=None, test_path=None):
    """
    Train and evaluate a Gemini classifier model.
    
    Args:
        model_class: The model class to use
        config: Configuration dictionary
        train_path: Path to training data
        test_path: Path to test data
        
    Returns:
        Tuple: (model, metrics, predictions)
    """
    if config is None:
        raise ValueError("Configuration dictionary must be provided")
        
    if train_path is None or test_path is None:
        train_path, test_path = load_cross_validation_path(config)
    
    # Set seed for reproducibility
    seed = config.get("seed", SEED_TRAINING)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize the model
    model = model_class(config)
    
    # Load data
    train_df, val_df, test_df = model.load_data(train_path, test_path)
    
    # Train and evaluate
    metrics, predictions = model.train_and_evaluate(
        train_df, val_df, test_df,
        epochs=config.get("epochs", 1)
    )
    
    return model, metrics, predictions