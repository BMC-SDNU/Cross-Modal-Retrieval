require 'nn'
require 'nngraph'
require 'cunn'
require 'cutorch'
require 'cudnn'
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  dataset = '',
  batchSize = 1,          -- number of samples to produce
  txtSize = 300,          -- dim for input text features
  imgSize = 4096,         -- dim for input image features
  embed_size = 1024,      -- dim for embeded features
  test_set_size = 100,    -- size of test set
  gpu = 2,                -- gpu mode. 0 = CPU, 1 = GPU
  data_root = '',
  checkpoint_dir = '',
  net_IG = '',
  net_TG = '',
  iter = 0
}

for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.display == 0 then opt.display = false end

if  opt.gpu > 0 then
  cutorch.setDevice(opt.gpu)
end

print(opt.net_IG)
print(opt.net_TG)
net_IG = torch.load(opt.checkpoint_dir .. '/' .. opt.net_IG)
net_TG = torch.load(opt.checkpoint_dir .. '/' .. opt.net_TG)
net_IG:evaluate()
net_TG:evaluate()

local input_txt = torch.Tensor(opt.test_set_size,opt.txtSize)
local input_img = torch.Tensor(opt.test_set_size,opt.imgSize)
test_txt_t7=torch.load(opt.data_root .. '/test_txt.t7') 
test_img_t7=torch.load(opt.data_root .. '/test_img.t7')
input_txt=test_txt_t7.test_txt
input_img=test_img_t7.test_img
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  net_IG:cuda()
  net_TG:cuda()
end

txt_common_rep=torch.Tensor(opt.test_set_size,opt.embed_size)
img_common_rep=torch.Tensor(opt.test_set_size,opt.embed_size)

for i=1,opt.test_set_size do
  local cur_fea_txt = input_txt[i]
  local cur_fea_img = input_img[i]
  cur_fea_txt=cur_fea_txt:reshape(1,opt.txtSize)
  cur_fea_img=cur_fea_img:reshape(1,opt.imgSize)
  local output = net_TG:forward(cur_fea_txt:cuda())
  local output2 = net_IG:forward(cur_fea_img:cuda())
  txt_common_rep[i]=net_TG:get(1).output:float()
  img_common_rep[i]=net_IG:get(1).output:float()
end

matio=require 'matio'
matio.save('eval/results/txt_common_rep_'..opt.iter..'.mat',txt_common_rep)
matio.save('eval/results/img_common_rep_'..opt.iter..'.mat',img_common_rep)



