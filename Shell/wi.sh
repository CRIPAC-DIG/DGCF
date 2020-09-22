#!/user/bin/env/ bash
CUDA_VISIBLE_DEVICES=5 nohup  python2 -u  DGCF.py \
 --network wikipedia \
 --model jodie \
 --method attention \
 --epochs 50 \
 --span_num 500 \
 --adj \
 --embedding_dim 64 \
 >./results/wiki_40&

