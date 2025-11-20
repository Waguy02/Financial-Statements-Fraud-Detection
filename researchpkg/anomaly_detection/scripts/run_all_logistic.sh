

#DATASET VERSION V4
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_logistic.py
done


#DATASET VERSION V5
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_logistic.py
done