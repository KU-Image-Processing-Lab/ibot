./run.sh imagenet_pretrain "test" vit_large teacher 16 \
  --teacher_temp 0.07 \
  --warmup_teacher_temp_epochs 30 \
  --norm_last_layer false \
  --epochs 800 \
  --batch_size_per_gpu 64 \
  --shared_head true \
  --out_dim 8192 \
  --local_crops_number 10 \
  --global_crops_scale 0.25 1 \
  --local_crops_scale 0.05 0.25 \
  --pred_ratio 0 0.3 \
  --pred_ratio_var 0 0.2