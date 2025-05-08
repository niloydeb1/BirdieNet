#!/usr/bin/env bash
# run_eval.sh

DATA_DIR="/home/amolaei/CvT/FGV/ViT-FGVC8-main/stanford_dog/updated_scripts/sasha___birdsnap/default/0.0.0/37beb97b7900281cd67ac189d4fb91c589b25582"
WEIGHTS="/home/amolaei/CvT/FGV/ViT-FGVC8-main/stanford_dog/updated_scripts/birdsnap_weights/cvt24_b3_skv1_d0.5_fit2.th"
BATCH_SIZE=8
MAX_SAMPLES=39860
MODEL_TYPE="cvt"  #choices=['cvt','vit']
VIT_VARIANT="base"    #choices=['base','large']
CVT_VARIANT="large" #choices=['xsmall','small','large','auto']
OUTPUT_JSON="/home/amolaei/CvT/FGV/ViT-FGVC8-main/stanford_dog/updated_scripts/brid525_cvt_large_C.json"

CUDA_VISIBLE_DEVICES=6 python3 /home/amolaei/CvT/FGV/ViT-FGVC8-main/stanford_dog/updated_scripts/eval_bird525.py \
  --data_dir   "$DATA_DIR" \
  --weights    "$WEIGHTS" \
  --model_type "$MODEL_TYPE" \
  --cvt_variant "$CVT_VARIANT" \
  --vit_variant "$VIT_VARIANT" \
  --batch_size $BATCH_SIZE \
  --max_samples $MAX_SAMPLES \
  --output_json "$OUTPUT_JSON"