#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_all.py \
 --network wikipedia \
 --epoch 50 \
 --span_num 500 \
 --adj \
 --method attention \
 --length 100 \
 >./results/wiki_120_all_100&

