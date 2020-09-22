#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_all.py \
 --network reddit \
 --epoch 50 \
 --span_num 800 \
 --adj \
 --method attention \
 --length 120 \
 >./results/reddit_all_120len&

