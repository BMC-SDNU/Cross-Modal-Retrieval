require 'image'
dir = require 'pl.dir'

trainLoader = {}

-- save filenames according to classnames file
local classnames = {}
for line in io.lines(opt.classnames) do
  classnames[#classnames + 1] = line
end

local files = {}
local trainids = {}
local size = 0
for line in io.lines(opt.trainids) do
  local id = tonumber(line)
  local dirpath = opt.data_root .. '/' .. classnames[id]
  cur_files = dir.getfiles(dirpath)
  files[id] = cur_files
  size = size + #cur_files
  trainids[#trainids + 1] = id
end

-- read data in a batch
function trainLoader:sample(quantity)

  local ix_batch1 = torch.Tensor(quantity)
  local ix_batch2 = torch.Tensor(quantity)
  local ix_file1 = torch.Tensor(quantity)
  local ix_file2 = torch.Tensor(quantity)

  for n = 1, quantity do
    local cls_ix = torch.randperm(#trainids):narrow(1,1,2)
    local id1 = trainids[cls_ix[1]]
    local id2 = trainids[cls_ix[2]]
    local file_ix1 = torch.randperm(#files[id1])[1]
    local file_ix2 = torch.randperm(#files[id2])[1]
    ix_batch1[n] = cls_ix[1]
    ix_batch2[n] = cls_ix[2]
    ix_file1[n] = file_ix1
    ix_file2[n] = file_ix2
  end

  local real_img=torch.Tensor(quantity,opt.imgSize)
  local wrong_img=torch.Tensor(quantity,opt.imgSize)
  local real_txt=torch.Tensor(quantity,opt.txtSize)
  local wrong_txt=torch.Tensor(quantity,opt.txtSize)  
  local ids = torch.zeros(quantity)

  local txt_batch_ix = 1
  for n = 1, quantity do
    local id1 = trainids[ix_batch1[n]]
    local id2 = trainids[ix_batch2[n]]
    ids[n] = id1
    local cls1_files = files[id1]
    local cls2_files = files[id2]

    local t7file1 = cls1_files[ix_file1[n]]
    local t7file2 = cls2_files[ix_file2[n]]
    local info1 = torch.load(t7file1)
    local info2 = torch.load(t7file2)
   
    local img1=info1.img
    local img2=info2.img 
    real_img[n]:copy(img1)
    wrong_img[n]:copy(img2)
    
    for s = 1, opt.numCaption do
      local ix_txt1 = torch.randperm(info1.txt:size(1))[1]
      local ix_txt2 = torch.randperm(info2.txt:size(1))[1]
      real_txt[txt_batch_ix]:copy(info1.txt[ix_txt1])
      wrong_txt[txt_batch_ix]:copy(info2.txt[ix_txt2])
      txt_batch_ix = txt_batch_ix + 1
    end
     
  end
  collectgarbage(); collectgarbage()
  return real_img, real_txt, wrong_img, wrong_txt, ids
end

function trainLoader:size()
  return size
end

