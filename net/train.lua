require "io"
require "gnuplot"
require "net-toolkit"
require "nn"
require "torch"
require "optim"
require "xlua"

cmd = torch.CmdLine()
cmd:text("Train a MNIST classifier")
cmd:text()
cmd:text("Options")
cmd:option("-model", "cnn", "Model type, cnn or mlp")
cmd:option("-eta", 0.1, "Learning rate")
cmd:option("-bsize", 5, "Batch size")
cmd:option("-epochs", 4, "Training epochs")
args = cmd:parse(arg)

print("nn: prepare model")

classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} -- 10 is the label for 0

if args.model == "cnn" then
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
elseif args.model == "mlp" then
    inputs = 28 * 28
    hidden = 100
    outputs = #classes

    model = nn.Sequential()
    model:add(nn.Reshape(inputs))
    model:add(nn.Linear(inputs, hidden))
    model:add(nn.ReLU())
    model:add(nn.Linear(hidden, outputs))
else
    error("nn: unknown model name \"" .. args.model .. "\"")
end

criterion = nn.CrossEntropyCriterion() -- LogSoftMax with ClassNLLCriterion

params, grad_params = model:getParameters()
optim_config = { learningRate = args.eta }
epochs = args.epochs
batch_size = args.bsize

print("nn: prepare training and testing datasets")

mnist = require "mnist"

mnist_train = mnist.traindataset()
x_train = mnist_train.data / 255
y_train = mnist_train.label
train_size = x_train:size()[1]

mnist_test = mnist.testdataset()
x_test = mnist_test.data / 255
y_test = mnist_test.label
test_size = x_test:size()[1]

print("nn: train model: " .. args.model)

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
batch_index = 0
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

model_name = string.format(
    "%s_model-%s.t7",
    args.model,
    os.date("%y%m%d_%H%M%S")
)
netToolkit.saveNet(model_name, model)
print("nn: model saved as " .. model_name)
