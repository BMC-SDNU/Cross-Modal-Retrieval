require 'nn'
local LSTM = require 't2i_attention.modules.LSTM'

local layer, parent = torch.class('nn.lstm', 'nn.Module')
function layer:__init(rnn_size, num_layers, dropout, seq_length)
    parent.__init(self)
    self.rnn_size = rnn_size
    self.num_layers = num_layers
    local dropout = dropout
    self.seq_length = seq_length
    
    self.core = LSTM.lstm(self.rnn_size, self.rnn_size, self.num_layers, dropout)
    
    self:_createInitState(1)
    self.core_output = torch.Tensor()
end

function layer:_createInitState(batch_size)
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size)
    end
  end
  self.num_state = #self.init_state
  end

function layer:createClones()
    print('constructing clones inside the sen_seq_level')
    self.cores = {self.core}
    for t=1,self.seq_length do
        self.cores[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:shareClones()
    if self.cores == nil then self:createClones(); return; end
    print('resharing clones inside the sen_seq_level')
    self.cores[1] = self.core
    for t=1,self.seq_length do
        self.cores[t]:share(self.core, 'weight', 'bias', 'gradWeight', 'gradBias')
    end
end

function layer:getModulesList()
    return {self.core}
end

function layer:parameters()
    -- we only have two internal modules, return their params
    local p1,g1 = self.core:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end

    return params, grad_params
end


function layer:training()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:training() end
end

function layer:evaluate()
    if self.cores == nil then self:createClones() end -- create these lazily if needed
    for k,v in pairs(self.cores) do v:evaluate() end
end

function layer:updateOutput(input)
  local sen_seq = input

  if self.cores == nil then self:createClones() end -- lazily create clones on first forward pass
  local batch_size = sen_seq:size(1)

  self:_createInitState(batch_size)
  self.fore_state = {[0] = self.init_state}
  self.fore_inputs = {}
  self.core_output:resize(batch_size, self.seq_length, self.rnn_size):zero()

  for t=1,self.seq_length do
      self.fore_inputs[t] = {sen_seq:narrow(2,t,1):contiguous():view(-1, self.rnn_size), unpack(self.fore_state[t-1])}
      local out = self.cores[t]:forward(self.fore_inputs[t])
      self.fore_state[t] = {} -- the rest is state
      for i=1,self.num_state do table.insert(self.fore_state[t], out[i]) end
      
      self.core_output:narrow(2,t,1):copy(out[self.num_state+1])
  end

  return self.core_output
end

function layer:updateGradInput(input, gradOutput)
  local sen_seq = input

  local batch_size = sen_seq:size(1)

  -- go backwards and lets compute gradients
  local d_core_state = {[self.seq_length] = self.init_state} -- initial dstates
  local d_embed_core = d_embed_core or self.core_output:new()
  d_embed_core:resize(batch_size, self.seq_length, self.rnn_size):zero()

  for t=self.seq_length,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#d_core_state[t] do table.insert(dout, d_core_state[t][k]) end
    table.insert(dout, gradOutput:narrow(2,t,1):contiguous():view(-1, self.rnn_size))
    local dinputs = self.cores[t]:backward(self.fore_inputs[t], dout)

    d_core_state[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do table.insert(d_core_state[t-1], dinputs[k]) end
    d_embed_core:narrow(2,t,1):copy(dinputs[1])
  end
  self.gradInput = d_embed_core
  return self.gradInput
end
