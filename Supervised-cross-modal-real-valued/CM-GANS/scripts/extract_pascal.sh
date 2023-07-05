ITER=2000
checkpoint_dir="checkpoints" \
data_root="data/test_data" \
dataset="pascal" \
net_IG="ckpt_pascal_${ITER}_net_IG.t7" \
net_TG="ckpt_pascal_${ITER}_net_TG.t7" \
iter=${ITER} \
th extract_feature.lua