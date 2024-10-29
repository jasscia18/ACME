NODE_RANK=0
NUM_GPUS=1

outdir=../Output1/SOON/pretrain/


CUDA_VISIBLE_DEVICES='0' python3  train_soon_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/soon_obj_model_config.json \
    --config config/soon_obj_pretrain.json \
    --output_dir $outdir