require 'torch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'cutorch'
require 'image'
require 'paths'
require 'optim'

local lstm_train = require 'txt-model.lstm_train'

--file name of images
file = io.open("/home/zhang/my-work/coco_train_file/train_image_list.txt",'r') 
--label indicator matrix for all data
label = io.open("/home/zhang/my-work/coco_train_file/train_label_mat.txt",'r')

--image model:deep residual network(50 layers)
model = torch.load('/home/zhang/my-work/resnet/resnet-50.t7') 
model = model:cuda()
model:remove(11) --remove the last layer of resnet

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

-- dict
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
lookup_table_1 = nn.LookupTable(vocab_size, input_size)
function extract_one_hot(input) -- input is a sentence
    word_list = mysplit(input, ' ')
    -- 
    input_list = {}
    for word_ = 1, 8 do
        num_idx = dict_number[word_list[word_]]
        table.insert(input_list, num_idx)
    end
    input_list_long = torch.LongTensor(input_list)
    output = lookup_table_1:forward(input_list_long)
    return output, input_list_long
end

--total number of paired samples: 82081
total_num = 82081
train_num = 0
label_index = torch.Tensor(total_num,81)
file_name = {}
for label_line in label:lines() do
    train_num = train_num + 1
    tmp_vector = mysplit(label_line,' ')
    table.insert(file_name,tmp_vector[1])
    for index = 1, #tmp_vector-1 do
        tmp = tonumber(tmp_vector[index+1])
        label_index[train_num][index] = tmp
    end
end 

loss_tmp = {}
table.insert(loss_tmp,0)
for i = 1, total_num do
    table.insert(loss_tmp,1/i)
end

loss_table = {}
table.insert(loss_table,0)
for j = 2, #loss_tmp do
    loss_table[j] = loss_tmp[j] + loss_table[j-1]
end

----optim parameters---------
method = optim.sgd
optimState={
learningRate=0.001,
momentum=0.9,
weightDecay=0.00005,}

local optim_state = {}
local optim_state1 = {}
local optim_state2 = {}

opt.epsilon = 1e-6
opt.alpha = 0.4
opt.learning_rate = 1e-1--learning rate

-----------parameters on feature dimension---------
seq_length = 8
img_dim = 2048
txt_dim = 512
emb_dim = 128

--img_w: image transformation; txt_w: text transformation
img_w=nn.Sequential():add(nn.Linear(img_dim,emb_dim)) -- Image mapping matrix
txt_w=nn.Sequential():add(nn.Mean(1)):add(nn.Linear(txt_dim,emb_dim)):add(nn.Reshape(emb_dim,1))-- Text mapping matrix

--orthogonalization 
img_w_q,img_w_r = torch.qr(img_w:get(1).weight:t():double())
txt_w_q,txt_w_r = torch.qr(txt_w:get(2).weight:t():double())

img_w:get(1).weight = img_w_q:t():cuda()
txt_w:get(2).weight = txt_w_q:t()

-- normalization for mappingsss
img_norm = torch.norm(img_w:get(1).weight)
txt_norm = torch.norm(txt_w:get(2).weight)
img_w:get(1).weight = img_w:get(1).weight/img_norm
txt_w:get(2).weight = txt_w:get(2).weight/txt_norm

img_w = img_w:cuda()
img_w_p,img_w_g = img_w:getParameters() --Get the parameters of the image matrix
img_p,img_g = model:getParameters()  --Get the parameters of the image model
txt_w_p,txt_w_g = txt_w:getParameters() 
look_up_p, look_up_g = lookup_table_1:getParameters()


train_num  = 0
for line in file:lines() do
 	train_num = train_num + 1
    print('Processing sample ID', train_num)	
--------------read a pair of positive image and its corresponding text-----------------
 	pos_img_train = image.load('/home/zhang/datasets/coco/train2014/'..line):mul(255):floor()
    pos_img_train = image.scale(pos_img_train,224,224,"bicubic")
    if (pos_img_train:size()[1] == 1) then
        pos_train = torch.CudaTensor(3,pos_img_train:size()[2],pos_img_train:size()[3])
        for i = 1, 3 do
            pos_train[i] = pos_img_train:clone()
        end
        pos_img_train = pos_train:clone():double()
    end
	pos_img_train = pos_img_train:clone():resize(1,3,224,224)
    --Using the ResNet extract the image feature 
    pos_img_fea = model:forward(pos_img_train:cuda()):clone()
    pos_img_fea = (pos_img_fea/torch.norm(pos_img_fea)):clone()
    
	--find the positive text 
    line_tmp = string.gsub(line,'.jpg','.txt')
    file1 = io.open("/home/zhang/my-work/coco_train_captions/"..line_tmp,'r')
    
	for line1 in file1:lines() do
		pos_txt_train, pos_input_list = extract_one_hot(line1)
        pos_txt_train = pos_txt_train:clone():resize(1,pos_txt_train:size(1),pos_txt_train:size(2))
        if (pos_txt_train:size(2) >= 8) then
            --using LSTM model extract text feature
            pos_txt_fea1, pos_rnn_state1 = lstm_train.forward(pos_txt_train, seq_length, 1)
            pos_txt_fea = torch.CudaTensor(seq_length, txt_dim)
            for i = 1, seq_length do
                pos_txt_fea[i] = pos_txt_fea1[i]:clone():resize(txt_dim)
                pos_txt_fea[i] = (pos_txt_fea[i]/torch.norm(pos_txt_fea[i])):clone()
            end
            pos_rnn_state = {}
            for j = 0, seq_length do
                pos_rnn_state[j] = {}
                for i=1, 2 do table.insert(pos_rnn_state[j], pos_rnn_state1[j][i]:clone():resize(1, 512)) end 
            end
            break
        end
	end

    --The ground truth of the above paired samples
    label_true = label_index[train_num]

    --The process of bi-directional sampling
    pos_img_map=img_w:forward(pos_img_fea:cuda()):clone()
    pos_txt_map=txt_w:forward(pos_txt_fea:double()):clone()

    --The distance of true paired samples
    true_dis = torch.sum(torch.cmul(pos_img_map,pos_txt_map:cuda()))
    print('Before: distance between positive image and positive text', true_dis)

------------------------image query texts----------------------------------------
    img_puni = 0   
    img_flag = 1
    while img_flag == 1 do
        local txt_index = math.random(total_num)
        local label_com = label_index[txt_index]
        local com = (label_true == label_com)
        --Find the inter-class samples 
        if com == false then
            img_puni = img_puni + 1
            local puni_txt_name = file_name[txt_index] 
            local txt_name = string.gsub(puni_txt_name,'.jpg','.txt')
            file2 = io.open("/home/zhang/my-work/coco_train_captions/"..txt_name,'r')
            for line2 in file2:lines() do
                puni_txt_train, puni_input_list = extract_one_hot(line2)
                puni_txt_train = puni_txt_train:clone():resize(1,puni_txt_train:size(1),puni_txt_train:size(2))
                if (puni_txt_train:size(2) >= 8) then
                    puni_txt_fea1, puni_rnn_state1 = lstm_train.forward(puni_txt_train, seq_length, 1)
                    puni_txt_fea = torch.CudaTensor(seq_length,txt_dim)
                    for i=1,seq_length do
                        puni_txt_fea[i] = puni_txt_fea1[i]:clone():resize(txt_dim)
                        puni_txt_fea[i] = (puni_txt_fea[i]/torch.norm(puni_txt_fea[i])):clone()
                    end
                    puni_rnn_state = {}
                    for j = 0, seq_length do
                        puni_rnn_state[j] = {}
                        for i=1, 2 do table.insert(puni_rnn_state[j], puni_rnn_state1[j][i]:clone():resize(1, 512)) end 
                    end
                    break
                end
            end
            puni_txt_map = txt_w:forward(puni_txt_fea:double()):clone()
            local tmp_dis = torch.sum(torch.cmul(pos_img_map,puni_txt_map:cuda()))
            if 0.3 + tmp_dis > true_dis then
                print('Before: distance between positive image and negative text', tmp_dis)
                img_flag = 0 
            end
        end
    end

--------------------------------------------------------------------------------------    
---------------------------------text query images-----------------------------------
    txt_puni = 0
    txt_flag = 1
    while txt_flag == 1 do
        local img_index = math.random(total_num)
        local label_com = label_index[img_index]
        local com = (label_true == label_com)
        if com == false then 
            txt_puni = txt_puni + 1
            local tmp_img = file_name[img_index]
            puni_img_train = image.load('/home/zhang/datasets/coco/train2014/'..tmp_img):mul(255):floor()
            puni_img_train = image.scale(puni_img_train,224,224,"bicubic")
            
            if (puni_img_train:size()[1] == 1) then
                puni_train = torch.CudaTensor(3,puni_img_train:size()[2],puni_img_train:size()[3])
                for i = 1, 3 do
                    puni_train[i] = puni_img_train:clone()
                end
                puni_img_train = puni_train:clone():double()
            end

            puni_img_train = puni_img_train:clone():resize(1,3,224,224)
            puni_img_fea = model:forward(puni_img_train:cuda()):clone()

            puni_img_fea = (puni_img_fea/torch.norm(puni_img_fea)):clone()

            puni_img_map = img_w:forward(puni_img_fea:cuda()):clone()
            local tmp_dis = torch.sum(torch.cmul(puni_img_map,pos_txt_map:cuda()))
            if 0.3 + tmp_dis > true_dis then
                print('Before: distance between negative image and positive text', tmp_dis)
                txt_flag = 0
            end
        end
    end

----------------------------------------------------------------------------------------
    --Each time find the quadruple, we calculate the gradient
    obj = (loss_table[torch.floor(total_num/img_puni)]/loss_table[total_num])*(0.3-torch.mm(pos_img_map,pos_txt_map:cuda()) + torch.mm(pos_img_map,puni_txt_map:cuda()))
        + (0.5-torch.mm(pos_img_map,pos_img_map:t()) + torch.mm(pos_img_map,puni_img_map:t()))
        + (loss_table[torch.floor(total_num/txt_puni)]/loss_table[total_num])*(0.3-torch.mm(puni_img_map,pos_txt_map:cuda()) + torch.mm(puni_img_map,pos_txt_map:cuda()))
        + (0.5-torch.mm(pos_txt_map:cuda():t(),pos_txt_map:cuda()) + torch.mm(puni_txt_map:cuda():t(),pos_txt_map:cuda()))
    print('Value of obejective function', obj)

    --Image gradient:positive image gradient and negative image gradient
    img_grad_pos = loss_table[torch.floor(total_num/img_puni)]*(-pos_txt_map+puni_txt_map)+loss_table[torch.floor(total_num/txt_puni)]*(-pos_txt_map)-pos_img_map:t():double()+puni_img_map:t():double()
    img_grad_neg = loss_table[torch.floor(total_num/txt_puni)]*(pos_txt_map)+pos_img_map:t():double()

    --Text gradient: positive text gradient and negative text gradient
    txt_grad_pos = loss_table[torch.floor(total_num/img_puni)]*(-pos_img_map)+loss_table[torch.floor(total_num/txt_puni)]*(-pos_img_map+puni_img_map)-pos_txt_map:cuda()+puni_txt_map:cuda()
    txt_grad_neg = loss_table[torch.floor(total_num/img_puni)]*(pos_img_map)+pos_txt_map:cuda()

    --backpropagation
    d_pos_txt_fea = txt_w:backward(pos_txt_fea:double(), txt_grad_pos:double()):clone()
    d_neg_txt_fea = txt_w:backward(puni_txt_fea:double(), txt_grad_neg:double()):clone()
    pos_look = lstm_train.backward(pos_txt_train, d_pos_txt_fea:cuda(), pos_rnn_state):clone()

    lookup_table_1:backward(pos_input_list, pos_look):clone()
    neg_look = lstm_train.backward(puni_txt_train, d_neg_txt_fea:cuda(), puni_rnn_state):clone()
   
    lookup_table_1:backward(puni_input_list, neg_look):clone()

---------------------update parameters for deep resnet----------------------------
    function feval(x)
        if(torch.cat(img_w_p,img_p,1)~=x) then 
            img_w_p=x[{{1,img_w_p:size()[1]}}]
            img_p=x[{{1+img_w_p:size()[1],-1}}]
        end
        feature=model:forward(torch.cat(pos_img_train,puni_img_train,1):cuda()):clone()
        u=img_w:forward(feature)
        alpha_u= torch.cat(img_grad_pos:t(),img_grad_neg:t(),1)
        img_w:zeroGradParameters()
        feature_g=img_w:backward(feature,alpha_u:cuda()):clone()
        model:zeroGradParameters()
        model:backward(torch.cat(pos_img_train,puni_img_train,1):cuda(),feature_g):clone()
        --img_w_g:add(img_map_grad)
        return obj,torch.cat(img_w_g,img_g,1)
    end

    function rmsprop(x, dx, lr, alpha, epsilon, state)
      if not state.m then
        state.m = x.new(#x):zero()
        state.tmp = x.new(#x)
      end
      -- calculate new (leaky) mean squared values
      state.m:mul(alpha)
      state.m:addcmul(1.0-alpha, dx, dx)
      -- perform update
      state.tmp:sqrt(state.m):add(epsilon)
      x:addcdiv(-lr, dx, state.tmp)
    end

    method(feval,torch.cat(img_w_p,img_p,1),optimState)--Stochastic Gradient Descent

 ---------------update------------------------------------------------------------------
    --update parameters for image mapping
    -- rmsprop(img_w_p,img_w_g,opt.learning_rate, opt.alpha, opt.epsilon, optim_state)

    --update parameters for text mapping
    rmsprop(txt_w_p,txt_w_g,opt.learning_rate, opt.alpha, opt.epsilon, optim_state1)

    rmsprop(look_up_p,look_up_g,opt.learning_rate, opt.alpha, opt.epsilon, optim_state2)
-----------------------------------------------------------------------------------------

----------------------orthogonal constraint to mapping matrices------------------------
    img_w_q,img_w_r = torch.qr(img_w:get(1).weight:t():double())
    txt_w_q,txt_w_r = torch.qr(txt_w:get(2).weight:t():double())

    img_w:get(1).weight = img_w_q:t():cuda()
    txt_w:get(2).weight = txt_w_q:t()

--------------------------------------------------------------------------------------

----------------------constrain mapping matrix to 1 ------------------------------------
    img_norm = torch.norm(img_w:get(1).weight)
    txt_norm = torch.norm(txt_w:get(2).weight)

    img_w:get(1).weight = img_w:get(1).weight/img_norm
    txt_w:get(2).weight = txt_w:get(2).weight/txt_norm

---------------------------------------------------------------------------------------
    -- distances after mapping 
    aft_img_map=img_w:forward(pos_img_fea:cuda()):clone()
    aft_txt_map=txt_w:forward(pos_txt_fea:double()):clone()
    --The distance of true paired samples
    aft_true_dis = torch.sum(torch.cmul(aft_img_map,aft_txt_map:cuda()))
    print('After: distance between positive image and positive text', aft_true_dis)

    after_txt = txt_w:forward(puni_txt_fea:double()):clone()
    tmp_dis1 = torch.sum(torch.cmul(aft_img_map,after_txt:cuda()))
    print('After: distance between positive image and negative text', tmp_dis1)

    after_img = img_w:forward(puni_img_fea:cuda()):clone()
    tmp_dis2 = torch.sum(torch.cmul(after_img,aft_txt_map:cuda()))
    print('after: distance between negative image and positive text', tmp_dis2)

    if (train_num == 82081) then
        torch.save('./coco_checkpoint/img_model.t7', model) --resnet model
        mapping = {}
        mapping.img = img_w
        mapping.txt = txt_w
        mapping.lookuptable = lookup_table_1
        torch.save('./coco_checkpoint/mapping_model.t7', mapping) --image and text mapping model
        lstm_model = {}
        lstm_model.model = lstm_train.protos
        lstm_model.opt = opt
        torch.save('./coco_checkpoint/txt_model.t7', lstm_model)
        break
    end
end

