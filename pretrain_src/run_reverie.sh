NODE_RANK=0
NUM_GPUS=1


outdir=../Output1/REVERIE/pretrain/

CUDA_VISIBLE_DEVICES='1' python3   train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_obj_pretrain.json \
    --output_dir $outdir