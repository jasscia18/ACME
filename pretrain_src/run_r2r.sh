NODE_RANK=0
NUM_GPUS=1
outdir=../Output1/R2R5/pretrain/


CUDA_VISIBLE_DEVICES='0' python3  train_r2r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r2r_pretrain.json \
    --output_dir $outdir

