require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'cutorch'

require 'image'
require 'paths'
require 'optim'

--load the trained model 
--image model:deep residual network(50 layers)
img_model = torch.load('/home/zhang/my-work/coco_checkpoint/img_model.t7')
--text model: lstm (2 layers)
txt_model = torch.load('/home/zhang/my-work/coco_checkpoint/txt_model.t7')
opt = txt_model.opt -- initiate parameters for LSTM model
--the mapping matrices
mapping = torch.load('/home/zhang/my-work/coco_checkpoint/mapping_model.t7')

--file name
file = io.open("/home/zhang/my-work/coco_val_file/val_image_list.txt",'r')
--label indicator matrix, each row is a label vector
label_vec = io.open("/home/zhang/my-work/coco_val_file/val_label_mat.txt",'r')


function mysplit(inputstr, sep)
    if sep == nil then
            sep = "%s" 
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str
        i = i + 1
    end
    return t
end

--validation: total number of paired samples: 40137
total_num = 40137
val_num = 0
label_index = torch.Tensor(total_num,81)
file_name = {}
for label_line in label_vec:lines() do
    val_num = val_num + 1
    tmp_vector = mysplit(label_line,' ')
    table.insert(file_name,tmp_vector[1])
    for index = 1, #tmp_vector-1 do
        tmp = tonumber(tmp_vector[index+1])
        label_index[val_num][index] = tmp
    end
end 

function read_dict()
    path = './coco_train_file/coco_dic.txt'
    local file = io.open(path, 'r')
    if file then print("File exist") end
    for line in file:lines() do
        dict_ = mysplit(line, ' ')
        vocab_size = #dict_
    end
    dict_number = {}
    for data_ = 1, #dict_ do
        dict_number[dict_[data_]] = data_
    end

    return dict_number
end

dict_number = read_dict()

input_size = 512
function extract_one_hot(input) -- input is a sentence
    word_list = mysplit(input, ' ')
    --lookup_table_1 = mapping.lookuptable(vocab_size, input_size)
    input_list = {}
    for word_ = 1, 8 do
        num_idx = dict_number[word_list[word_]]
        table.insert(input_list, num_idx)
    end
    input_list_long = torch.LongTensor(input_list)
    --output = lookup_table_1:forward(input_list_long)
    output = mapping.lookuptable:forward(input_list_long)
    return output
end

-- init the cell/hidden states
init_state = {}
for L = 1, opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size):cuda()
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

function preprocessed(x)
    -- swap the axis for fast training
    x = x:transpose(1,2):contiguous() 
    x = x:float():cuda()
    return x
end

function extract_lstm_fea(txt_mat, length, max_batches)
    local rnn_state = {[0] = init_state}
    local fea = {}
    for i = 1, max_batches do
        x = txt_mat:clone()
        x = preprocessed(x)
        txt_model.model.rnn:evaluate()
        for t = 1, length do
            lst = txt_model.model.rnn:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1, #init_state do table.insert(rnn_state[t], lst[i]) end
            table.insert(fea, lst[#lst]:clone())
        end
    end
    return fea
end

--Extract features for each image and text
total_test_num = 5000
label_num = 81
seq_length = 8
txt_dim = 512
emb_dim = 128
test_num  = 0

low_img = torch.CudaTensor(total_test_num, emb_dim)
low_txt = torch.CudaTensor(total_test_num, emb_dim)
label_test = torch.CudaTensor(total_test_num,label_num)

img_deep_fea = {}
txt_deep_fea = {}

for line in file:lines() do
    test_num = test_num + 1
    print('Test samples', test_num)
    print(line)
    --read the positive image
    img_test = image.load('/home/zhang/datasets/coco/val2014/'..line):mul(255):floor()
    img_test = image.scale(img_test,224,224,"bicubic")

    if (img_test:size()[1] == 1) then
        pos_val = torch.CudaTensor(3,img_test:size()[2],img_test:size()[3])
        for i = 1, 3 do
            pos_val[i] = img_test
        end
        img_test = pos_val:double()
    end

    img_test = img_test:resize(1,3,224,224):clone()
    --Using the ResNet extract the image feature 
    img_fea = img_model:forward(img_test:cuda()):clone()

    img_fea = (img_fea/torch.norm(img_fea)):clone()

    table.insert(img_deep_fea,img_fea)

    --find its corresponding text 
    line_tmp = string.gsub(line,'.jpg','.txt')
    txt_cap = io.open("/home/zhang/my-work/coco_val_captions/"..line_tmp,'r')
    
    for line1 in txt_cap:lines() do
        txt_val = extract_one_hot(line1)
        txt_val:resize(1,txt_val:size(1),txt_val:size(2))
        if (txt_val:size(2) >= 8) then
            --using LSTM extract text feature
            txt_fea1 = extract_lstm_fea(txt_val,seq_length,1)
            txt_fea = torch.CudaTensor(seq_length, txt_dim)
            for i = 1, seq_length do
                txt_fea[i] = txt_fea1[i]:clone():resize(txt_dim)
                txt_fea[i] = (txt_fea[i]/torch.norm(txt_fea[i])):clone()
            end
            break
        end
    end
    -- print(txt_fea)
    table.insert(txt_deep_fea,torch.mean(txt_fea,1))
   

    label_true = label_index[test_num]
    label_test[test_num] = label_true

    --obtain the mapped features for each paired samples
    img_map=mapping.img:forward(img_fea:cuda()):clone()
    txt_map=mapping.txt:forward(txt_fea:double()):clone()

    low_img[test_num] = img_map
    low_txt[test_num] = txt_map

    if (test_num == 5000) then
        deep_feature = {}
        deep_feature.img = img_deep_fea
        deep_feature.txt = txt_deep_fea
        deep_feature.label = test_label
        torch.save('./coco_checkpoint/coco_val_feature.t7', deep_feature)
        break
    end
end

--Compute mean average precision
--y: true label
--ybar: predicted label
function AveragePrecision(y, ybar, k)
    ap = 0
    num_relevant = 0
    pred, idx = torch.sort(ybar,true)

    for i = 1, k do
        if (y[idx[i]] > 0) then
            num_relevant = num_relevant + 1
            ap = ap + num_relevant/i
        end
    end

    if (num_relevant == 0) then
        ap = 0
    else
        if (num_relevant >= k) then
            ap = ap/k
        else
            ap = ap/num_relevant
        end
    end
    return ap
end

ytrue = torch.mm(label_index, label_index:t())

ally = torch.mm(low_img,low_txt:t())
allyhat = torch.mm(label_test,label_test:t())

--MAP score of image-query-texts
img_Map = {}
img_Map1 = {}
sum_img = 0
sum_img1 = 0
test_num1 = 50
for i = 1, total_test_num do
    y = ally[i]
    yhat = allyhat[i]
    img_map = AveragePrecision(yhat,y,test_num)
    img_map1 = AveragePrecision(yhat,y,test_num1)
    sum_img = sum_img + img_map
    sum_img1 = sum_img1 + img_map1
    table.insert(img_Map, img_map)
    table.insert(img_Map1, img_map1)
end
imgMap = sum_img/total_test_num
imgMap1 = sum_img1/total_test_num
print('MAP scores of image-query-texts in 5000 is', imgMap)
print('MAP scores of image-query-texts in 50 is', imgMap1)

-- MAP score of image-query-texts
ally = ally:t()
txt_Map = {}
txt_Map1 = {}
sum_txt = 0
sum_txt1 = 0
for i = 1, total_test_num do
    y = ally[i]
    yhat = allyhat[i]
    txt_map = AveragePrecision(yhat,y,test_num)
    txt_map1 = AveragePrecision(yhat,y,test_num)
    sum_txt = sum_txt + txt_map
    sum_txt1 = sum_txt1 + txt_map1
    table.insert(txt_Map, txt_map)
    table.insert(txt_Map1, txt_map1)
end
txtMap = sum_txt/total_test_num
txtMap1 = sum_txt1/total_test_num
print('MAP scores of txt-query-images in 5000 is', txtMap)
print('MAP scores of txt-query-images in 50 is', txtMap1)
