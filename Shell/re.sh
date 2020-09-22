#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  DGCF.py \
 --network reddit\
 --model jodie \
 --span_num 1000 \
 --epochs 50 \
 --method attention \
 --adj \
 --length 120\
 >./results/re_120&

