DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=../Output1/R2R5/finetune-vis1/

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 15
      --max_instr_len 200

      --batch_size 1
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adam

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0.

      "


# train
CUDA_VISIBLE_DEVICES='0' PYTHONIOENCODING=utf-8 python3 main_r2r.py $flag  \
      --tokenizer bert \
      --eval_first \
      --bert_ckpt_file ../Output1/R2R5/pretrain/ckpts/model_step_177500.pt \

