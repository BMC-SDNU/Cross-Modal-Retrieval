require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

opt = {
  numCaption = 1,
  save_every = 100,
  print_every = 1,
  dataset = '',
  data_root = '',
  classnames = '',
  trainids = '',
  checkpoint_dir = 'checkpoints',
  batchSize = 64,
  txtSize = 300,            -- dim for input text features
  imgSize = 4096,           -- dim for input image features
  embed_size = 1024,      -- dim for embeded features
  nThreads = 4,           -- data loading threads to use
  niter = 1000,           
  lr = 0.001,             
  lr_decay = 0.5,         
  decay_every = 100,      
  beta1 = 0.5,            -- momentum term of adam
  ntrain = math.huge,     -- examples per epoch. math.huge for full dataset
  gpu = 0,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
  name = '',
  init_TG = '',
  init_TD = '',
  init_IG = '',
  init_ID = '',
  init_CD_I = '',
  init_CD_T = '',
  use_cudnn = 0,
  class_num = 20,
}
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

if opt.gpu > 0 then
   ok, cunn = pcall(require, 'cunn')
   ok2, cutorch = pcall(require, 'cutorch')
   cutorch.setDevice(opt.gpu)
end

opt.manualSeed = torch.random(1, 10000)
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- initialization------------------------------------------------------------
-- create data loader
local DataLoader = paths.dofile('data/data.lua')
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("Dataset: " .. opt.dataset, " Size: ", data:size())
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

-- Network Architecture------------------------------------------------------
local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution
-- text generator
if opt.init_TG == '' then 

  encoder_txt1 = nn.Sequential()
  encoder_txt1:add(nn.Linear(opt.txtSize,1024)) 
  encoder_txt1:add(nn.BatchNormalization(1024))
  encoder_txt1:add(nn.LeakyReLU(0.2,true))
  
  encoder_txt2 = nn.Sequential()
  encoder_txt2:add(nn.Linear(1024,opt.embed_size)) 
  encoder_txt2:add(nn.BatchNormalization(opt.embed_size))
  encoder_txt2:add(nn.LeakyReLU(0.2,true))
  
  encoder_txt = nn.Sequential()
  encoder_txt:add(encoder_txt1)
  encoder_txt:add(encoder_txt2)
  
  decoder_txt = nn.Sequential()
  decoder_txt:add(nn.Linear(opt.embed_size,1024))
  decoder_txt:add(nn.BatchNormalization(1024))
  decoder_txt:add(nn.LeakyReLU(0.2,true))
  decoder_txt:add(nn.Linear(1024,opt.txtSize))
  
  classifier_txt = nn.Sequential()
  classifier_txt:add(nn.Linear(opt.embed_size, opt.class_num))
  classifier_txt:add(nn.LogSoftMax())
  
  dc_txt = nn.ConcatTable()
  dc_txt:add(classifier_txt)
  dc_txt:add(decoder_txt)
  
  netTG = nn.Sequential()
  netTG:add(encoder_txt)
  netTG:add(dc_txt)
  netTG:apply(weights_init)
else
  netTG = torch.load(opt.init_TG)
end

-- image generator
if opt.init_IG == '' then

  encoder_img1 = nn.Sequential()
  encoder_img1:add(nn.Linear(opt.imgSize,1024)) 
  encoder_img1:add(nn.BatchNormalization(1024))
  encoder_img1:add(nn.LeakyReLU(0.2,true))
  
  encoder_img2 = nn.Sequential()
  encoder_img2:add(nn.Linear(1024,opt.embed_size)) 
  encoder_img2:add(nn.BatchNormalization(opt.embed_size))
  encoder_img2:add(nn.LeakyReLU(0.2,true))
  
  -- share weights with encoder_txt2
  encoder_img2:share(encoder_txt2,'weight','bias','gradWeight','gradBias')
  
  encoder_img = nn.Sequential()
  encoder_img:add(encoder_img1)
  encoder_img:add(encoder_img2)
  
  decoder_img = nn.Sequential()
  decoder_img:add(nn.Linear(opt.embed_size,1024))
  decoder_img:add(nn.BatchNormalization(1024))
  decoder_img:add(nn.LeakyReLU(0.2,true))
  decoder_img:add(nn.Linear(1024,opt.imgSize))
  
  classifier_img = nn.Sequential()
  classifier_img:add(nn.Linear(opt.embed_size, opt.class_num))
  classifier_img:add(nn.LogSoftMax())
  
  dc_img = nn.ConcatTable()
  dc_img:add(classifier_img)
  dc_img:add(decoder_img)
  
  netIG = nn.Sequential()
  netIG:add(encoder_img)
  netIG:add(dc_img)
  netIG:apply(weights_init)
else
  netIG = torch.load(opt.init_IG)
end

-- text discriminator
if opt.init_TD == '' then
  netTD = nn.Sequential()
  netTD:add(nn.Linear(opt.txtSize,1))
  netTD:add(nn.Sigmoid())
  netTD:add(nn.View(1))
  netTD:apply(weights_init)
else
  netTD = torch.load(opt.init_TD)
end

-- image discriminator
if opt.init_ID == '' then
  netID = nn.Sequential()
  netID:add(nn.Linear(opt.imgSize,1))
  netID:add(nn.Sigmoid())
  netID:add(nn.View(1))
  netID:apply(weights_init)
else
  netID = torch.load(opt.init_ID)
end

if opt.init_CD_I == '' then
  ipt = nn.ParallelTable()
  ipt:add(nn.Identity())
  ipt:add(nn.Identity())

  netCD_I = nn.Sequential()
  netCD_I:add(ipt)
  netCD_I:add(nn.JoinTable(2))
  netCD_I:add(nn.Linear(opt.embed_size+opt.imgSize,512))
  netCD_I:add(nn.BatchNormalization(512))
  netCD_I:add(nn.LeakyReLU(0.2, true))
  netCD_I:add(nn.Linear(512,1))
  netCD_I:add(nn.Sigmoid())
  netCD_I:add(nn.View(1))
  netCD_I:apply(weights_init)
else
  netCD_I = torch.load(opt.init_CD_I)
end

if opt.init_CD_T == '' then
  ipt = nn.ParallelTable()
  ipt:add(nn.Identity())
  ipt:add(nn.Identity())

  netCD_T = nn.Sequential()
  netCD_T:add(ipt)
  netCD_T:add(nn.JoinTable(2))
  netCD_T:add(nn.Linear(opt.embed_size+opt.txtSize,512))
  netCD_T:add(nn.BatchNormalization(512))
  netCD_T:add(nn.LeakyReLU(0.2, true))
  netCD_T:add(nn.Linear(512,1))
  netCD_T:add(nn.Sigmoid())
  netCD_T:add(nn.View(1))
  netCD_T:apply(weights_init)
else
  netCD_T = torch.load(opt.init_CD_T)
end

-- LOSS---------------------------------------------------------------------
-- binary cross-entropy loss for GAN
local criterion = nn.BCECriterion()

-- L1 loss which forces features generated by TG and IG to be closer
local criterionAE = nn.AbsCriterion()

-- classification loss
local criterionClass = nn.ClassNLLCriterion()
assert(math.floor(opt.batchSize / opt.numCaption) * opt.numCaption == opt.batchSize)

-- DATA---------------------------------------------------------------------
local real_label = 1
local fake_label = 0
local img_label = 1
local txt_label = 0

local input_img = torch.Tensor(opt.batchSize, opt.imgSize)
local input_txt = torch.Tensor(opt.batchSize, opt.txtSize)
local input_wrong_img = torch.Tensor(opt.batchSize, opt.imgSize)
local input_wrong_txt = torch.Tensor(opt.batchSize, opt.txtSize)
local label = torch.Tensor(opt.batchSize) --real or false

local errID,errTD,errCD_I,errCD_T, errIG, errTG
local errL1 = 0

local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()
local ids = torch.Tensor(opt.batchSize)

if opt.gpu > 0 then
   input_img = input_img:cuda()
   input_txt = input_txt:cuda() 
   input_wrong_img = input_wrong_img:cuda()
   input_wrong_txt = input_wrong_txt:cuda() 
   label = label:cuda()
   ids = ids:cuda()
  
   netID:cuda()
   netIG:cuda()
   netTG:cuda()
   netTD:cuda()
   netCD_I:cuda()
   netCD_T:cuda()
   criterion:cuda()
   criterionAE:cuda()
   criterionClass:cuda()
end

if opt.use_cudnn == 1 then
  cudnn = require('cudnn')
  netID = cudnn.convert(netID, cudnn)
  netIG = cudnn.convert(netIG, cudnn)
  netTG = cudnn.convert(netTG, cudnn)
  netTD = cudnn.convert(netTD, cudnn)
  netCD_I = cudnn.convert(netCD_I, cudnn)
  netCD_T = cudnn.convert(netCD_T, cudnn)
end

local parametersTD, gradParametersTD = netTD:getParameters()
local parametersID, gradParametersID = netID:getParameters()
local parametersCD_I, gradParametersCD_I = netCD_I:getParameters()
local parametersCD_T, gradParametersCD_T = netCD_T:getParameters()
local parametersTG, gradParametersTG = netTG:getParameters()
local parametersIG, gradParametersIG = netIG:getParameters()

-- get data in batches
local sample = function() 
  data_tm:reset(); data_tm:resume()
  real_img,real_txt,wrong_img,wrong_txt,ids= data:getBatch()
  data_tm:stop()
  input_img:copy(real_img)
  input_txt:copy(real_txt)
  input_wrong_img:copy(wrong_img)
  input_wrong_txt:copy(wrong_txt)
end

-- forward and backward-----------------------------------------------------
local fTDx = function(x) 
  gradParametersTD:zero()
  
  local txt_common_recon = netTG:forward(input_txt)
  local output = netTD:forward(txt_common_recon[2])
  label:fill(fake_label)
  local errTD_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netTD:backward(txt_common_recon[2], df_do)
  
  local output = netTD:forward(input_txt)
  label:fill(real_label)
  local errTD_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netTD:backward(input_txt, df_do)
  
  errTD = errTD_fake + errTD_real

  return errTD, gradParametersTD
end

local fIDx = function(x) 
  gradParametersID:zero()
  
  local img_common_recon = netIG:forward(input_img)
  local output = netID:forward(img_common_recon[2])
  label:fill(fake_label)
  local errID_fake = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netID:backward(img_common_recon[2], df_do)
  
  local output = netID:forward(input_img)
  label:fill(real_label)
  local errID_real = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netID:backward(input_img, df_do)
  
  errID = errID_fake + errID_real

  return errID, gradParametersID
end

local fCD_Ix = function(x) 
  gradParametersCD_I:zero()
  
  local txt_common = encoder_txt:forward(input_txt)
  label:fill(fake_label)
  local output = netCD_I:forward({txt_common,input_img})
  local errCD_txt = criterion:forward(output, label) * 0.5
  local df_do = criterion:backward(output, label) * 0.5
  netCD_I:backward({txt_common,input_img}, df_do)
  
  local img_common = encoder_img:forward(input_img)
  label:fill(real_label)
  local output = netCD_I:forward({img_common,input_img})
  local errCD_img = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netCD_I:backward({img_common,input_img}, df_do)

  local img_common_wrong = encoder_img:forward(input_wrong_img)
  label:fill(fake_label)
  local output = netCD_I:forward({img_common_wrong,input_wrong_img})
  local errCD_img_wrong = criterion:forward(output, label) * 0.5
  local df_do = criterion:backward(output, label) * 0.5
  netCD_I:backward({img_common_wrong,input_wrong_img}, df_do)

  errL1 = criterionAE:forward(txt_common, img_common)
  
  errCD_I = errCD_img + errCD_txt + errCD_img_wrong

  return errCD_I, gradParametersCD_I
end

local fCD_Tx = function(x) 
  gradParametersCD_T:zero()
  
  local img_common = encoder_img:forward(input_img)
  label:fill(fake_label)
  local output = netCD_T:forward({img_common,input_txt})
  local errCD_img = criterion:forward(output, label) * 0.5
  local df_do = criterion:backward(output, label) * 0.5
  netCD_T:backward({img_common,input_txt}, df_do)
  
  local txt_common = encoder_txt:forward(input_txt)
  label:fill(real_label)
  local output = netCD_T:forward({txt_common,input_txt})
  local errCD_txt = criterion:forward(output, label)
  local df_do = criterion:backward(output, label)
  netCD_T:backward({txt_common,input_txt}, df_do)

  local txt_common_wrong = encoder_txt:forward(input_wrong_txt)
  label:fill(fake_label)
  local output = netCD_T:forward({txt_common_wrong,input_wrong_txt})
  local errCD_txt_wrong = criterion:forward(output, label) * 0.5
  local df_do = criterion:backward(output, label) * 0.5
  netCD_T:backward({txt_common_wrong,input_wrong_txt}, df_do)

  errL1 = criterionAE:forward(txt_common, img_common)
  
  errCD_T = errCD_img + errCD_txt + errCD_txt_wrong

  return errCD_T, gradParametersCD_T
end

local fTGx = function(x)
  gradParametersTG:zero()

  local txt_common = encoder_txt:forward(input_txt)
  label:fill(real_label)
  local output = netCD_I:forward({txt_common,input_img})
  local errCD_txt = criterion:forward(output, label)
  local df_cd = criterion:backward(output, label)
  local d_cd = netCD_I:backward({txt_common,input_img}, df_cd)
  
  local txt_common_recon = dc_txt:forward(txt_common)
  local output = netTD:forward(txt_common_recon[2])
  label:fill(real_label)
  local errTD_txt = criterion:forward(output, label)
  local df_td = criterion:backward(output, label)
  local d_td = netTD:backward(txt_common_recon[2], df_td)
  
  local errCL_txt = criterionClass:forward(txt_common_recon[1], ids)
  local d_cl = criterionClass:backward(txt_common_recon[1], ids)
  local d_de = dc_txt:backward(txt_common, {d_cl, d_td})
  
  encoder_txt:backward(input_txt, d_cd[1]+d_de)
  
  errTG = errCD_txt + errTD_txt + errCL_txt
  
  return errTG, gradParametersTG
end

local fIGx = function(x)
  gradParametersIG:zero()

  local img_common = encoder_img:forward(input_img)
  label:fill(real_label)
  local output = netCD_T:forward({img_common,input_txt})
  local errCD_img = criterion:forward(output, label)
  local df_cd = criterion:backward(output, label)
  local d_cd = netCD_T:backward({img_common,input_txt}, df_cd)

  local img_common_recon = dc_img:forward(img_common)
  local output = netID:forward(img_common_recon[2])
  label:fill(real_label)
  local errID_img = criterion:forward(output, label)
  local df_id = criterion:backward(output, label)
  local d_id = netID:backward(img_common_recon[2], df_id)
 
  local errCL_img = criterionClass:forward(img_common_recon[1], ids)
  local d_cl = criterionClass:backward(img_common_recon[1], ids)
  local d_de = dc_img:backward(img_common, {d_cl, d_id})
  
  encoder_img:backward(input_img, d_cd[1]+d_de)
  
  errIG = errCD_img + errID_img + errCL_img
  
  return errIG, gradParametersIG
end

optimStateIG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateTG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateID = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateTD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateCD_I = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateCD_T = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

-- train--------------------------------------------------------------------
for epoch = 1, opt.niter do
  epoch_tm:reset()

  if epoch % opt.decay_every == 0 then
    optimStateIG.learningRate = optimStateIG.learningRate * opt.lr_decay
    optimStateTG.learningRate = optimStateTG.learningRate * opt.lr_decay
    optimStateID.learningRate = optimStateID.learningRate * opt.lr_decay
    optimStateTD.learningRate = optimStateTD.learningRate * opt.lr_decay
    optimStateCD_I.learningRate = optimStateCD_I.learningRate * opt.lr_decay
    optimStateCD_T.learningRate = optimStateCD_T.learningRate * opt.lr_decay
  end

  for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
    tm:reset()
	sample()
    
    optim.adam(fTDx, parametersTD, optimStateTD)
    optim.adam(fIDx, parametersID, optimStateID)
    optim.adam(fCD_Tx, parametersCD_T, optimStateCD_T)
    optim.adam(fCD_Ix, parametersCD_I, optimStateCD_I)
	for m=1,5 do 
        optim.adam(fIGx, parametersIG, optimStateIG)
  
        optim.adam(fTGx, parametersTG, optimStateTG)        
	end
    
    -- logging
    if ((i-1) / opt.batchSize) % opt.print_every == 0 then
      print(('[%d][%d/%d] T:%.3f  DT:%.3f lr: %.4g '
                .. '  Err_IG: %.4f  Err_ID: %.4f  Err_TG: %.4f  Err_TD: %.4f  Err_CD: %.4f L1: %.4f'):format(
              epoch, ((i-1) / opt.batchSize),
              math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize),
              tm:time().real, data_tm:time().real,
              optimStateIG.learningRate,
              errIG and errIG or -1, errID and errID or -1,errTG and errTG or -1, errTD and errTD or -1,errCD_I and errCD_I or -1, errL1))
    end
  end
  
  -- saving
  if epoch % opt.save_every == 0 then
    paths.mkdir(opt.checkpoint_dir)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_IG.t7', netIG)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_ID.t7', netID)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_TG.t7', netTG)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_TD.t7', netTD)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_CD_I.t7', netCD_I)
    torch.save(opt.checkpoint_dir .. '/' .. opt.name .. '_' .. epoch .. '_net_CD_T.t7', netCD_T)
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
           epoch, opt.niter, epoch_tm:time().real))
  end
end

