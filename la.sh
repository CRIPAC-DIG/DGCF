#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  DGCF.py \
 --network lastfm \
 --model jodie \
 --epochs 50 \
 --method attention \
 --adj \
 --span_num 500 \
 --length 20 \
 >./results/la_20&

