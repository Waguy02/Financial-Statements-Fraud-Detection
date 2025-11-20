## Dataset V4: CIK SPLIT

# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_llama8b.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_mda_llama8b.py
done

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_mda_llama8b.py
done

## Dataset V5: Random Split
# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_fin_llama8b.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_mda_llama8b.py
done    

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_fin_mda_llama8b.py
done


## Dataset V4 : Zero Shot

# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_llama8b_zero_shot.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_mda_llama8b_zero_shot.py
done

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_mda_llama8b_zero_shot.py
done







