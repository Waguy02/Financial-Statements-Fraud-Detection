
#DATASET VERSION V4
for fold_id in 1 2 3 4 5; do
    DATASET_VERSION="company_isolated_splitting" FOLD_ID=$fold_id python v4_mlp.py
done


#DATASET VERSION V5
for fold_id in 1 2 3 4 5; do
    DATASET_VERSION="time_splitting" FOLD_ID=$fold_id python v5_mlp.py
done