require 'i2t_attention.modules.lstm_level'

local HybridCNNLSTM = {}

function HybridCNNLSTM.cnn(batch_size, alphasize, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 256

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  -- 2904 x alphasize
  net:add(nn.TemporalConvolution(alphasize, 384, 15))
  net:add(nn.View(384):setNumInputDims(2))
  net:add(nn.BatchNormalization(384))
  net:add(nn.View(2890,384):setNumInputDims(2))
  net:add(nn.ReLU())
  net:add(nn.TemporalMaxPooling(5, 5))
  -- 578 x 256
  net:add(nn.TemporalConvolution(384, 512, 9))
  net:add(nn.View(512):setNumInputDims(2))
  net:add(nn.BatchNormalization(512))
  net:add(nn.View(570,512):setNumInputDims(2))
  net:add(nn.ReLU())
  net:add(nn.TemporalMaxPooling(5, 5))
  -- 114 x 256
  net:add(nn.TemporalConvolution(512, cnn_dim, 7))
  net:add(nn.View(cnn_dim):setNumInputDims(2))
  net:add(nn.BatchNormalization(cnn_dim))
  net:add(nn.View(108,cnn_dim):setNumInputDims(2))
  net:add(nn.ReLU())
  net:add(nn.TemporalMaxPooling(6, 6))
  -- 20 x 256
  local h1 = net(inputs[1])

  local r2 = nn.lstm(cnn_dim, 2, 0.5, 18)(h1)
  out = nn.Linear(cnn_dim, emb_dim)(nn.Dropout(dropout)(nn.View(cnn_dim):setNumInputDims(2)(r2)))
  out = nn.View(18,emb_dim):setNumInputDims(2)(out)
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return HybridCNNLSTM

