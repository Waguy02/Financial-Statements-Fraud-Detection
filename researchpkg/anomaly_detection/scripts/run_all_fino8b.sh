## Dataset V4: company-isolated split

# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_fino8b.py
done


# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_mda_fino8b.py
done


# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_mda_fino8b.py
done


## Dataset V5: Random Split

# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_fin_fino8b.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_mda_fino8b.py
done    

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v5_fin_mda_fino8b.py
done


## Dataset: Zero-shot

# FIN
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_fino8b_zero_shot.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_mda_fino8b_zero_shot.py
done

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    FOLD_ID=$fold_id python v4_fin_mda_fino8b_zero_shot.py
done








# V4 FIN MDA DECHOW AND EXTENDED
for fold_id in 1 2 3 4 5; do
    run_jz h100 "FOLD_ID=$fold_id OFFLINE=1 FORCE_RANDOM_SPLIT=1" 4 python v4_fin_mda_fino8b.py
done