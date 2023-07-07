gpu=0
data_path=./data
vocab_path=./data/vocab/

noise_ratio=0.2
# if you want load a pre-index file, pls set noise_file
noise_file=noise_file_path

# noise_ratio:[0.2,0.4,0.6,0.8]---warmup_epoch:[2,2,1,1]
warmup_epoch=2 
module_name=SGR
folder_name=coco_SGR_noise0.2

python  train.py --gpu $gpu --warmup_epoch $warmup_epoch --folder_name $folder_name --noise_ratio $noise_ratio --noise_file $noise_file --num_epochs 20 --lr_update 10 --module_name $module_name --learning_rate 0.0005 --data_name coco_precomp --data_path $data_path  --vocab_path $vocab_path
