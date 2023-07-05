-- Necessary functionalities
require 'nn'
require 'nngraph'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'i2t_attention.modules.lstm_level'
require 'hdf5'

matio = require 'matio'

local model_utils = require('i2t_attention.util.model_utils')

--cutorch.setDevice(1)

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
--local test_txt = matio.load(path.join(opt.data_dir, 'test_txt.mat')).test_txt

test_image = test_image:float():cuda()
local img_fea = protos.enc_image:cuda():forward(test_image)

local f = hdf5.open(path.join(opt.data_dir, 'wiki_txt.hdf5'), 'r')
local w2v = f:read('w2v'):all()
local txt_data = f:read('test'):all()
local lookup = nn.LookupTable(76480, 300)
lookup.weight:copy(w2v)
lookup.weight[1]:zero()
txt_mat = lookup:forward(txt_data)

txt_fea = torch.zeros(txt_mat:size(1), model.opt.emb_dim)
local step = 63
local it = txt_mat:size(1) / step

for i=1,it do
  local test_txt_one = torch.zeros(step,txt_mat:size(2),300)
  test_txt_one[{{},{},{}}] = txt_mat[{{(i-1)*step+1,i*step},{},{}}]
  local txt_RNN_fea = protos.enc_doc:forward(test_txt_one:float():cuda())
  local txt_atten = protos.attention:forward(txt_RNN_fea)
  txt_fea[{{(i-1)*step+1,i*step},{}}] = txt_atten:float()
end

feature_dir = './i2t_attention/extracted_feature/'
if not path.exists(feature_dir) then lfs.mkdir(feature_dir) end
img_fea_path = feature_dir .. 'img_fea.mat'
txt_fea_path = feature_dir .. 'txt_fea.mat'
matio.save(img_fea_path, img_fea:float())
matio.save(txt_fea_path, txt_fea:float())
