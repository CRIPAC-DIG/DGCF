#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  evaluate_interaction_prediction.py \
 --network wikipedia \
 --epoch 48 \
 --method attention \
 --adj \
 --length 120 \
 >./results/wiki_neig120_48_test&

