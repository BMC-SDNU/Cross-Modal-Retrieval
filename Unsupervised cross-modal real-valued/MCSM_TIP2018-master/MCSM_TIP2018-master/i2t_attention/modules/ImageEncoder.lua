
local ImageEncoder = {}
function ImageEncoder.enc(input_size, hid_size, noop)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- Image
  local x = inputs[1]
  local outputs = {}

  if noop == 1 then
      table.insert(outputs, x)
  else
      h1 = nn.Linear(input_size, hid_size)(x)
      sig1 = nn.Sigmoid()(h1)
      sig2 = nn.Normalize(2)(sig1)
      table.insert(outputs, sig2)
  end
  return nn.gModule(inputs, outputs)
end

return ImageEncoder

