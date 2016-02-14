require "io"
require "gnuplot"
require "net-toolkit"
require "nn"
require "torch"
require "optim"
require "xlua"

mnist = require "mnist"

print("nn: prepare training dataset")

mnist_train = mnist.traindataset()
x_train = mnist_train.data / 255
y_train = mnist_train.label

print("nn: prepare test dataset")

mnist_test = mnist.testdataset()
x_test = mnist_test.data / 255
y_test = mnist_test.label

print("nn: prepare model")

classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} -- 10 is the label for 0

model = nn.Sequential()
model:add(nn.Reshape(1, 28, 28))
-- 1
model:add(nn.SpatialConvolution(1, 16, 5, 5))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(3, 3))
-- 2
model:add(nn.SpatialConvolution(16, 64, 3, 3))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2))
-- 3
model:add(nn.Reshape(64 * 3 * 3))
model:add(nn.Linear(64 * 3 * 3, 100))
model:add(nn.ReLU())
model:add(nn.Linear(100, #classes))

criterion = nn.CrossEntropyCriterion() -- LogSoftMax with ClassNLLCriterion

params, grad_params = model:getParameters()
optim_config = {
    learningRate = 0.1,
}

epochs = 4
batch_size = 5
batch_index = 0

train_size = x_train:size()[1]
test_size = x_test:size()[1]

print("nn: train model")

function feval(x)
    if x ~= params then
        params:copy(x)
    end

    local batch_start = batch_index * batch_size + 1
    local batch_end = math.min(train_size, (batch_index + 1) * batch_size + 1)
    batch_index = batch_end == train_size and 0 or batch_index + 1

    local batch_input = x_train[{{batch_start, batch_end}, {}}]
    local batch_target = y_train[{{batch_start, batch_end}}]

    grad_params:zero()
    local batch_output = model:forward(batch_input)
    local batch_loss = criterion:forward(batch_output, batch_target)
    local grad_output = criterion:backward(batch_output, batch_target)
    model:backward(batch_input, grad_output)

    return batch_loss, grad_params
end

function ftest()
    local conf = optim.ConfusionMatrix(classes)
    conf:zero()
    local iterations = math.ceil(test_size / batch_size) - 1
    for i = 0, iterations do
        local batch_start = i * batch_size + 1
        local batch_end = math.min(test_size, (i + 1) * batch_size + 1)
        local batch_input = x_test[{{batch_start, batch_end}, {}}]
        local batch_target = y_test[{{batch_start, batch_end}}]
        conf:batchAdd(model:forward(batch_input), batch_target)
    end
    return conf
end

iterations = epochs * math.ceil(train_size / batch_size)
updates = math.ceil(0.1 * iterations)
for i = 1, iterations do
    optim.adagrad(feval, params, optim_config)
    if i % updates == 0 then
        model:evaluate()
        io.write("nn: Testing ")
        print(ftest())
        model:training()
    end
    xlua.progress(i, iterations)
end

print("nn: save model")

name = "cnn_model.t7"
netToolkit.saveNet(name, model)
print("nn: model saved as " .. name)
