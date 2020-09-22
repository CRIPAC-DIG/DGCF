#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_all.py \
 --network lastfm \
 --epoch 50 \
 --span_num 500 \
 --adj \
 --method attention \
 --length 60 \
 >./results/la_all_60len&

