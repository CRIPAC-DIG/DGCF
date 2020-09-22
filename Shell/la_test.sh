#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_interaction_prediction.py \
 --network lastfm \
 --method attention \
 --epoch 21 \
 --adj \
 >./results/last_21_atten_test&

