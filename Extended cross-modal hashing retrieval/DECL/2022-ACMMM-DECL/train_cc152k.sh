gpu=0
data_path=./data
vocab_path=./data/vocab/

warmup_epoch=5
module_name=SGR
folder_name=cc152k_SGR

python  train.py --gpu $gpu --warmup_epoch $warmup_epoch --folder_name $folder_name --noise_ratio 0 --num_epochs 40 --lr_update 20 --module_name $module_name --learning_rate 0.0002 --data_name cc152k_precomp --data_path $data_path  --vocab_path $vocab_path
