for dataset_version in v4 v5; do
    for feature_type in EXTENDED DECHOW EXTENDED_DECHOW; do
        export MODEL_CLASS="XGB" 
        export DATASET_VERSION=$dataset_version
        export FEATURES_TYPE=$feature_type
        export MAX_EVALS=50
        export CUDA_VISIBLE_DEVICES=0 
        python tune_hparams_financial_model.py
    done
done
