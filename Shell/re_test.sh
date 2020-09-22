#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_interaction_prediction.py \
 --network reddit \
 --epoch 30 \
 --span_num 1000 \
 --adj \
 --method attention \
 >./results/reddit_attention_30&

