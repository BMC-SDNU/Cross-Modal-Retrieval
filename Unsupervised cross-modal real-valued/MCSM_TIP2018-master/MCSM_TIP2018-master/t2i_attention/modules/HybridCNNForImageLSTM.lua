require 't2i_attention.modules.lstm_level'

local HybridCNNForImageLSTM = {}
function HybridCNNForImageLSTM.cnn(image_dim, emb_dim, dropout, avg, cnn_dim)
  dropout = dropout or 0.0
  avg = avg or 0
  cnn_dim = cnn_dim or 256

  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Bag

  local net = nn.Sequential()
  local r2 = nn.lstm(image_dim, 2, 0.5, 49)(inputs[1])
  out = nn.Linear(image_dim, emb_dim)(nn.Dropout(dropout)(nn.View(image_dim):setNumInputDims(2)(r2)))
  out = nn.View(49,emb_dim):setNumInputDims(2)(out)
  local outputs = {}
  table.insert(outputs, out)
  return nn.gModule(inputs, outputs)
end

return HybridCNNForImageLSTM

