./run.sh polyp_pretrain "vit_large" vit_large teacher 8 \
  --data_path ../../dataset/imagenet/dataset/TrainDataset/images/0 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 50 \
  --weight_decay_end 0.48 \
  --norm_last_layer false \
  --epochs 250 \
  --batch_size_per_gpu 16 \
  --shared_head true \
  --out_dim 8192 \
  --local_crops_number 10 \
  --global_crops_scale 0.25 1 \
  --local_crops_scale 0.05 0.25 \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2 \
  --clip_grad 0.3 \
  --drop_path 0.2 \
  --freeze_last_layer 3 \
  --min_lr 0.0002 \
  --num_workers 4 \
  --lambda1 1.0

python3 ./extract_backbone_weights.py \
    ./work_dirs/vit_large/checkpoint0240.pth \
    ./work_dirs/vit_large/extracted_backbone_weights.pth \
    --checkpoint_key 'student' \
    --with_head false
