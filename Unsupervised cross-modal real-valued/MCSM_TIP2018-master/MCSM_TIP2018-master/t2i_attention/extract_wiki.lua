-- Necessary functionalities
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 't2i_attention.modules.lstm_level'

matio = require 'matio'

local model_utils = require('t2i_attention.util.model_utils')

cutorch.setDevice(1)

-- Encode query document using alphabet.
local alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
local dict = {}
for i = 1,#alphabet do
    dict[alphabet:sub(i,i)] = i
end

-------------------------------------------------
cmd = torch.CmdLine()
cmd:option('-data_dir','data','data directory.')
cmd:option('-image_dir','images','image subdirectory.')
cmd:option('-txt_dir','','text subdirectory.')
cmd:option('-savefile','sje_tcnn','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-symmetric',1,'symmetric sje')
cmd:option('-learning_rate',0.0001,'learning rate')
cmd:option('-testclasses', 'testclasses.txt', 'validation or test classes to be used in evaluation')
cmd:option('-ids_file', 'trainvalids.txt', 'file specifying which class labels were used for training.')
cmd:option('-model','','model to load. If blank then above options will be used.')
cmd:option('-txt_limit',0,'if 0 then use all available text. Otherwise limit the number of documents per class')
cmd:option('-num_caption',10,'number of captions per image to be used for training')
cmd:option('-ttype','char','word|char')
cmd:option('-gpuid',0,'gpu to use')

opt = cmd:parse(arg)
if opt.gpuid >= 0 then
  cutorch.setDevice(opt.gpuid+1)
end
local model
if opt.model ~= '' then
	model = torch.load(opt.model)
else
	model = torch.load(string.format('%s/lm_%s_%.5f_%.0f_%.0f_%s.t7', opt.checkpoint_dir, opt.savefile, opt.learning_rate, opt.symmetric, opt.num_caption, opt.ids_file))
end
-----------------------------------------------------------


local doc_length = model.opt.doc_length
local protos = model.protos
protos.enc_doc:evaluate()
protos.enc_image:evaluate()
protos.attention:evaluate()

local test_image = matio.load(path.join(opt.data_dir, 'test_img.mat')).test_img
local test_txt = matio.load(path.join(opt.data_dir, 'test_txt.mat')).test_txt

-- zcr
feature_dir = './t2i_attention/extracted_feature/'
if not path.exists(feature_dir) then lfs.mkdir(feature_dir) end
img_fea_path = feature_dir .. 'img_fea.mat'
txt_fea_path = feature_dir .. 'txt_fea.mat'

test_txt = test_txt:float():cuda()
local txt_fea = protos.enc_doc:forward(test_txt)
-- matio.save('txt_fea.mat', txt_fea:float())
matio.save(txt_fea_path, txt_fea:float())
print('text feature saving done!')

img_fea = torch.zeros(test_image:size(1), model.opt.emb_dim)
for i=1,test_image:size(1) do
  local test_image_one = torch.zeros(1,model.opt.img_seq_len,model.opt.image_dim)
  test_image_one[{1,{},{}}] = test_image[{i,{},{}}]
  local img_RNN_fea = protos.enc_image:forward(test_image_one:float():cuda())
  local img_atten = protos.attention:forward(img_RNN_fea)
  img_fea[{i,{}}] = img_atten:float()
end
-- matio.save('img_fea.mat', img_fea:float())
matio.save(img_fea_path, img_fea:float())
print('image feature saving done!')
