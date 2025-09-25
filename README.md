# Aligned Blankspace Augmentation (ABA) — Length Generalization

Minimal code to reproduce our results on length generalization for arithmetic and related algorithmic tasks using Aligned Blankspace Augmentation (ABA). Includes ABA-fixed / ABA-var, and several other baseline method. 

## Running Experiments:
python train.py config/addition_10digits.py \
--positional_embedding=RPEBias --max_iters=10000 \
--blank_space_in_equation_number=21 --digit_test_number=20 \
--blank_space_exact=True \
--out_dir out/addition_10 --wandb_run_name=run1 

python eval.py  \
  --out_dir out/addition_10/run1 \
  --ckpt out/addition_10/addition_10_best_ood.pt \
  --mode length --operator + --digit_test_number 20 --plot
