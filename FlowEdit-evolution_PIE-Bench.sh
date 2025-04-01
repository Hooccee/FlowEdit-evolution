python FlowEdit-evolution_batch.py \
  --device_number 0 \
  --dataset "PIE-Bench" \
  --height 512 \
  --width 512 \
  --model_type "FLUX" \
  --model_path "/data/chx/FLUX.1-dev" \
  --T_steps 28 \
  --tar_guidance_scale 5.5 \
  --output_dir my_results \
  --eval_metrics \
  --save_samples

#需要安装 torchvision pandas torchmetrics openpyxl