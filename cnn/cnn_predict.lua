require "net-toolkit"
require "nn"
require "torch"

model = netToolkit.loadNet("cnn_model.t7")
model:evaluate()

function predict(pixels)
    x = torch.Tensor(pixels) / 255

    pred = model:forward(x)
    pred = torch.exp(pred) -- LogSoftMax is used
    -- scale to between 0 and 1
    local pred_min = pred:min()
    local pred_max = pred:max()
    pred = (pred - pred_min) / (pred_max - pred_min)

    return pred:totable()
end

function predict_digit(pixels)
    local pred = predict(pixels)
    local _, digit = torch.max(pred, 1)
    digit = digit[1]
    return digit == 10 and 0 or digit
end
