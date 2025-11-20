
## Dataset V4 : Zero Shot

# FIN
for fold_id in 1 2 3 4 5; do
    run_jz a100 "FOLD_ID=$fold_id OFFLINE=1" 2 python v4_fin_qwen32b_zero_shot.py
done

# MDA
for fold_id in 1 2 3 4 5; do
    run_jz h100 "FOLD_ID=$fold_id OFFLINE=1" 4 python v4_mda_qwen32b_zero_shot.py
done

# FIN-MDA
for fold_id in 1 2 3 4 5; do
    run_jz h100 "FOLD_ID=$fold_id OFFLINE=1" 4 python v4_fin_mda_qwen32b_zero_shot.py
done







