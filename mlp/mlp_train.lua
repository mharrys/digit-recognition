require "nn"
require "torch"

-- prepare training dataset

mnist = require "mnist"
mnist_train = mnist.traindataset()

image = {
    w = 28,
    h = 28,
    pixels = 28 * 28
}
digits = 10

x = torch.reshape(mnist_train.data, mnist_train:size(), image.pixels) / 255
y = mnist_train.label

dataset = {}
function dataset:size() return mnist_train:size() end
for i = 1, dataset:size() do
    dataset[i] = {x[i], y[i]}
end

-- train model

inputs = image.pixels
hidden = 100
outputs = digits

model = nn.Sequential()
model:add(nn.Linear(inputs, hidden))
model:add(nn.ReLU())
model:add(nn.Linear(hidden, outputs))

criterion = nn.CrossEntropyCriterion() -- LogSoftMax with ClassNLLCriterion

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRateDecay = 0
trainer.learningRate = 0.01
trainer.maxIteration = 25
trainer:train(dataset)

-- prepare test dataset

mnist_test = mnist.testdataset()
x_test = torch.reshape(mnist_test.data, mnist_test:size(), image.pixels) / 255
y_test = mnist_test.label

-- test model

model:evaluate()
pred = model:forward(x_test)
errors = 0
for i = 1, mnist_test:size() do
    _, digit = torch.max(pred[i], 1)
    errors = digit[1] == y_test[i] and errors or errors + 1
end
print("mlp: " .. errors .. "/" .. mnist_test:size() .. " errors (" .. errors / mnist_test:size() .. ")")

-- save model

name = "mlp_model.t7"
torch.save(name, model)
print("mlp: model saved as " .. name)
