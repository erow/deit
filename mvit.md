# Multi-scale Vision Transformers



## train
```
# 4 gpus
WANDB_NAME=deit_base-baseline sbatch ~/storchrun.slurm main.py --model deit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/deit_base-baseline 



```

training mvit
```bash
export WANDB_NAME=mvit_base-n4-m0-scratch
sbatch -N 4 storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 3e-3 --epochs 400 --weight-decay 0.05 --sched cosine --input-size 192 --eval-crop-ratio 1.0 --reprob 0.0 --smoothing 0.0 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --use-wandb  --model_args mask_ratio=0.75 num_frames=4 
```
## finetune
```bash
export WANDB_NAME=mvit_base-n2-m0 
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0 num_frames=2 

export WANDB_NAME=mvit_base-n4-m0 
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0 num_frames=4 

export WANDB_NAME=mvit_base-n4-m0-384
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 384 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0 num_frames=4

export WANDB_NAME=mvit_base-n2-m0.5 
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0.5 num_frames=2 

export WANDB_NAME=mvit_base-n4-m0.5 
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0.5 num_frames=4 

export WANDB_NAME=mvit_base-n4-m0.75 
sbatch storchrun.slurm main.py --model mvit_base_patch16_LS --data-path $IMNET --batch 128 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --use-wandb --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --model_args mask_ratio=0.75 num_frames=4 
```

## larger size

```bash
# --nodes 1 --ngpus 8 = 8 gpus
export WANDB_NAME=mvit_large-n2-m0
sbatch --gpus=8 -n 2 -N 2 storchrun.slurm main.py --model mvit_large_patch16_LS --data-path $IMNET --batch 64 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.45 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --model_args mask_ratio=0 num_frames=2 --output_dir $SCRATCH/experiments/mvit/$WANDB_NAME --use-wandb

# --nodes 2 --ngpus 8 = 16 gpus
export WANDB_NAME=mvit_huge-n2-m0
sbatch --gpus=16 -n 4 -N 4 storchrun.slurm main.py --model mvit_huge_patch14_LS --data-path $IMNET --batch 32 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0 --nb-classes 1000 --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.55 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --pretrained --model_args mask_ratio=0 num_frames=2 --use-wandb

```