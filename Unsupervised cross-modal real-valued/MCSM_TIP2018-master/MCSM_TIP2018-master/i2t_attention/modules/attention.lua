require 'nngraph'
require 'nn'

local attention = {}
function attention.atten(input_size, embedding_size, seq_size)
    local inputs = {}
    local outputs = {}

    table.insert(inputs, nn.Identity()()) 

    local seq_feat = inputs[1]

    local atten_sum = nn.Dropout(0.5)(nn.Tanh()(seq_feat))
    local atten_embedding = nn.Linear(input_size, 1)(nn.View(input_size):setNumInputDims(2)(atten_sum))
    local atten = nn.SoftMax()(nn.View(seq_size):setNumInputDims(2)(atten_embedding))

    local atten_dim = nn.View(1,-1):setNumInputDims(1)(atten)
    local atten_feat = nn.MM(false, false)({atten_dim, seq_feat})
    atten_feat = nn.View(input_size):setNumInputDims(2)(atten_feat)
    atten_feat = nn.Normalize(2)(atten_feat)

    table.insert(outputs, atten_feat)
    
    return nn.gModule(inputs, outputs)
end

return attention
