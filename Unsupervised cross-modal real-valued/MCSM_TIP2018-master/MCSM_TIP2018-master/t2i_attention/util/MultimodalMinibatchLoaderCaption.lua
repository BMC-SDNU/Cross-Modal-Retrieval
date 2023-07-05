matio = require 'matio'

local model_utils = require('t2i_attention.util.model_utils')

local MultimodalMinibatchLoaderCaption = {}
MultimodalMinibatchLoaderCaption.__index = MultimodalMinibatchLoaderCaption

function MultimodalMinibatchLoaderCaption.create(data_dir, nclass, img_dim, doc_length,
                                                 batch_size, randomize_pair, ids_file, num_caption,
                                                 image_dir, flip, img_seq_len)
    local self = {}
    setmetatable(self, MultimodalMinibatchLoaderCaption)

    --[[self.alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    self.dict = {}
    for i = 1,#self.alphabet do
        self.dict[self.alphabet:sub(i,i)] = i
    end
    self.alphabet_size = #self.alphabet]]
    self.alphabet_size = 5000

    
    -- load manifest file.
    --[[self.files = {}
    for line in io.lines(path.join(data_dir, 'manifest.txt')) do
        self.files[#self.files + 1] = line
    end]]

    -- load train / val / test splits.
    --[[self.trainids = {}
    for line in io.lines(path.join(data_dir, ids_file)) do
        self.trainids[#self.trainids + 1] = line
    end
    --self.nclass_train = #self.trainids]]
    self.nclass_train = 2173
    self.trainids_tensor = torch.zeros(self.nclass_train)
    for i = 1,self.nclass_train do
        self.trainids_tensor[i] = i
    end
    self.nclass = nclass
    self.batch_size = batch_size
    self.data_dir = data_dir
    self.nclass = nclass
    self.img_dim = img_dim
    self.doc_length = doc_length
    self.ntrain = self.nclass
    self.randomize_pair = randomize_pair
    self.num_caption = num_caption
    self.image_dir = image_dir or ''
    self.flip = flip or 0
    self.img_seq_len = img_seq_len
    self.img_data = matio.load(path.join(self.data_dir, 'train_img.mat')).train_img

    collectgarbage()
    return self
end


function MultimodalMinibatchLoaderCaption:next_batch()
    local sample_ix = torch.randperm(self.nclass_train)
    sample_ix = sample_ix:narrow(1,1,self.batch_size)

    local txt = torch.zeros(self.batch_size, self.doc_length)
    local img = torch.zeros(self.batch_size, self.img_seq_len, self.img_dim)
    local lab = torch.zeros(self.batch_size)

    --local train_image = matio.load(path.join(self.data_dir, 'train_img.mat')).train_img
    local train_image = self.img_data
    local train_text = matio.load(path.join(self.data_dir, 'train_txt.mat')).train_txt
    local train_lab = matio.load(path.join(self.data_dir, 'train_lab.mat')).train_lab

    for i = 1,self.batch_size do
        local id = self.trainids_tensor[sample_ix[i]]
        --local fname = self.files[id]
	
        --[[local cls_imgs
        if (self.image_dir == '' or self.image_dir == nil) then
            cls_imgs = torch.load(path.join(self.data_dir, 'images', fname))
        else
            cls_imgs = torch.load(path.join(self.data_dir, self.image_dir, fname))
        end
	      local cls_sens = torch.load(path.join(self.data_dir, string.format('text_c%s', self.num_caption), fname))]]

        --local sen_ix = torch.Tensor(1)
        --sen_ix:random(1,cls_sens:size(3))

        --local ix = torch.randperm(cls_imgs:size(1))[1]
        --local ix = torch.randperm(cls_sens:size(1))[1]
        --local ix_view = torch.randperm(cls_imgs:size(3))[1]
        lab[i] = train_lab[id]
        --local img_id_list = torch.randperm(self.nclass_train)
        --local img_id = 0
        --for j=1,self.nclass_train do
        --    local tmp = train_lab[img_id_list[j]][1]
        --    if tmp == lab[i] then
        --        img_id = img_id_list[j]
        --        break
        --    end
        --end
        img[{i, {}, {}}] = train_image[{id, {}, {}}]

        txt[{i, {}}] = train_text[{id, {}}]
        
        --[[for j = 1,train_text:size(2) do
            local on_ix = train_text[{id, j}]
            
            --if cls_sens:size(1) == 1 then
            --	on_ix = cls_sens[{1, j, sen_ix[1]}]
            --else
            --	on_ix = cls_sens[{ix, j, sen_ix[1]}]
            --end            
            
            if on_ix == 0 then
                break
            end
            --if (self.flip == 1 and math.random() < 0.5) then
            if math.random() < self.flip then
                txt[{i, cls_sens:size(2) - j + 1, on_ix}] = 1
            else
                txt[{i, j, on_ix}] = 1
            end
        end]]
    end
    return txt, img, lab
end

return MultimodalMinibatchLoaderCaption

