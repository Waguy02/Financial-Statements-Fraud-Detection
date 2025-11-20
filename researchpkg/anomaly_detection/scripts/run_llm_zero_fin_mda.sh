for fold_id in  5
do
    echo "Running fold $fold_id"
    FOLD_ID=$fold_id python v4_fin_mda_llama70b_zero_shot.py
    FOLD_ID=$fold_id python v4_mda_llama70b_zero_shot.py
    echo "Fold $fold_id completed"
done