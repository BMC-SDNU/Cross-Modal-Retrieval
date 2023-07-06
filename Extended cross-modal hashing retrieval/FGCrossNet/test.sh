echo "run test"
CUDA_VISIBLE_DEVICES=0,1 python test.py --data_path='/path/dataset/' --snapshot='./model/rankingloss/model.pkl' --feature='./feature'