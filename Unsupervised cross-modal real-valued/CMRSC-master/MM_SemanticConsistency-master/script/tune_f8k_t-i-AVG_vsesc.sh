data_name=f8k_precomp
Batch_size=128
Attn=t-i
Loss=AVG
DATA_PATH=../../data
MODEL_DIR=checkpoint/f8k_scan/${Attn}_${Loss}

GPU_ID=1
mse_lambda=0.3

echo "---------------training--------------"
logger_name=${MODEL_DIR}/gpus_vsesc_${mse_lambda}
model_name=${logger_name}

echo ${logger_name}

python train_gpus.py --gpuid ${GPU_ID} --batch_size ${Batch_size} --data_path ${DATA_PATH} --data_name ${data_name} --vocab_path vocab --logger_name ${logger_name} --model_name ${model_name} --max_violation --bi_gru --agg_func=Mean --cross_attn=t2i --lambda_softmax=9 --model_mode vsesc --mse_lambda ${mse_lambda} --num_epochs=20 --lr_update=10

echo "---------------evaluation--------------"
MODEL_PATH=${model_name}/model_best.pth.tar
echo ${MODEL_PATH}

SPLIT=test

python test_gpus.py --gpuid ${GPU_ID} --model_path ${MODEL_PATH} --data_path ${DATA_PATH} --split ${SPLIT}
