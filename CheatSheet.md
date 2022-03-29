## For Google Colaboratory
```python
import sys, os

if 'google.colab' in sys.modules:
    # mount google drive
    from google.colab import drive
    drive.mount('/content/gdrive')
    path_to_file = '/content/gdrive/My Drive/CS4243_codes/codes/<FILE_PATH>'
    print(path_to_file)
    # move to Google Drive directory
    os.chdir(path_to_file)
    !pwd
```

## PyTorch Basic Knowledge
```python
import torch

# 1 x 3 vector with float values
x = torch.Tensor([5.3, 2.1, -3.1])
# 1 x 2 vector with long values
y = torch.LongTensor([5, 6])

x.type() # torch.FloatTensor
y.type() # torch.LongTensor
# float -> long
x = x.long()
# long -> float
y = y.float()
x.type() # torch.LongTensor
y.type() # torch.FloatTensor

# 2 x 2 matrix
A = torch.Tensor([[5.3, 2.1], [0.2, 2.1]])
# 10 x 2 random matrix
A = torch.rand(10, 2)
# 10 x 2 matrix filled with zeros
A = torch.zeros(10, 2)
# 5 x 2 x 2 random Tensor
B = torch.rand(5, 2, 2)

B.dim() # 3
B.type() # torch.FloatTensor
B.size() # torch.Size([5, 2, 2])
B.size(0) # 5
B.size(1) # 2
B.size(2) # 2

a = torch.arange(10) # tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
b = torch.randperm(10) # tensor([0, 4, 7, 9, 8, 5, 3, 1, 6, 2]) # random permutation

# 1 x 10 -> 2 x 5
a.view(2, 5)
# 1 x 10 -> 2 (auto conversion) x 5
a.view(-1, 5)
# 1 x 10 -> 5 x 2
c = a.view(5, 2)

d = c[0] # row 0
e = c[1] # row 1
f = c[1: 4] # row 1 to row 3
g = c[4, 0] # value in row 4 column 0, this is a scalar, not a tensor

# A matrix is 2-dimensional Tensor
# A row of a matrix is a 1-dimensional Tensor
# An entry of a matrix is a 0-dimensional Tensor
# 0-dimensional Tensor are scalar!
# If we want to convert a 0-dimensional Tensor into python number, we need to use item()
h = g.item()
type(g) # <class 'torch.Tensor'>
type(h) # <class 'int'>
```

```python
import utils

from utils import check_mnist_dataset_exists
data_path = check_mnist_dataset_exists()

# download mnist dataset
# 60,000 gray scale pictures as well as their label, each picture is 28 by 28 pixels
train_data = torch.load(data_path + 'mnist/train_data.pt')
train_label = torch.load(data_path + 'mnist/train_label.pt')
test_data = torch.load(data_path + 'mnist/test_data.pt')
test_label = torch.load(data_path + 'mnist/test_label.pt')

train_data.size() # torch.Size([60000, 28, 28])
train_label.size() # torch.Size([60000])
train_data[0] # picture 0
train_label[0] # picture 0 label
train_label[10000: 10000 + 5] # pictures 10000, 10001, 10002, 10003, 10004

test_data.size() # torch.Size([10000, 28, 28])
test_label.size() # torch.Size([10000])

utils.show(data[10]) # show picture 10
```

```python
import utils

from utils import check_cifar_dataset_exists
data_path = check_cifar_dataset_exists()

# download CIFAR dataset
# 50,000 RGB pictures as well as their label, each picture is 32 by 32 pixels with 3 color channels
train_data = torch.load(data_path + 'cifar/train_data.pt')
train_label = torch.load(data_path + 'cifar/train_label.pt')
test_data = torch.load(data_path + 'cifar/test_data.pt')
test_label = torch.load(data_path + 'cifar/test_label.pt')

train_data.size() # torch.Size([50000, 3, 32, 32])
train_label.size() # torch.Size([50000])

test_data.size() # torch.Size([10000, 3, 32, 32])
test_label.size() # torch.Size([10000])
```

```python
import torch
import torch.nn as nn

# Make a Linear Module that takes input of size 5 and return output of size 3
mod = nn.Linear(5, 3) # Linear(in_features=5, out_features=3, bias=True)
x = torch.rand(5) # 1 x 5 random vector
y = mod(x) # auto transpose x to 5 x 1 vector, output y is 1 x 3 vector (auto transpose)

mod.weight # 3 x 5 weights
mod.weight.size() # torch.Size([3, 5])

mod.bias # 1 x 3 biases (auto transpose)
mod.bias.size() # torch.Size([3])

# If we want we can change the internal parameters of the module:
with torch.no_grad():
    mod.weight[0, 0] = 0 # change row 0 column 0 weight to 0
    mod.weight[0, 1] = 1 # change row 0 column 1 weight to 1
    mod.weight[0, 2] = 2 # change row 0 column 2 weight to 2

# Make a Linear Module that takes input of size 5 and return output of size 3 without bias
mod2 = nn.Linear(5, 3, bias=False)
mod2.weight # 3 x 5 weights
mod2.weight.size() # torch.Size([3, 5])

mod2.bias # None
```

```python
import torch
import torch.nn as nn

# for vector
x = torch.Tensor([-2, -0.5, 2.0, 1.5]) # 1 x 4 vector
# dim=0 by default
sm1 = torch.softmax(x , dim=0) # tensor([0.0107, 0.0481, 0.5858, 0.3553]) 1 x 4 vector
sm1.sum() # tensor(1.)

# for matrix
A = torch.Tensor([[-2, -0.5, 2.0, 1.5], [-2, -0.5, 7.0, 1.5]]) # 2 x 4 matrix
# dim=1: 对每一行的所有元素进行softmax运算
sm2 = torch.softmax(A , dim=1)
sm2.sum(axis=1) # tensor([1., 1.])
# dim=0: 对每一列的所有元素进行softmax运算
sm3 = torch.softmax(A , dim=0)
sm3.sum(axis=0) # tensor([1., 1., 1., 1.])
```

```python
import torch
import torch.nn as nn

class two_layer_net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(two_layer_net , self).__init__()
        
        # 两层全连接网络MLP
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True)
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)        
        
    # 向前传播
    def forward(self, x):
        
        x = self.layer1(x)
        # 第一层使用ReLU作为激活函数
        x = torch.relu(x)
        x = self.layer2(x)
        # 第二层使用softmax作为激活函数并输出
        # dim=0或dim=1是由input决定的
        # 例如batch size = 200，则input是[[data_1], [data_2], ..., [data_200]]
        # 需要计算的是里面的维度的softmax，即data_1到data_200的softmax，则dim=1
        # 若没有batch size，input是data_1
        # 直接计算当前维度的softmax，则dim=0
        p = torch.softmax(x, dim=0)
        
        return p

# input_size = 2
# hidden_size = 5
# output_size = 3
net = two_layer_net(2, 5, 3)
net.layer1 # Linear(in_features=2, out_features=5, bias=True)
net.layer1.weight # 5 x 2 weights
net.layer1.bias # 1 x 5 biases (auto transpose)

x = torch.Tensor([1,1])
p = net.forward(x) # 向前传播一次
p = net(x) # 输出预测
p.sum() # tensor(1., grad_fn=<SumBackward0>)

with torch.no_grad():
    net.layer1.weight[0, 0] = 10 # change row 0 column 0 weight to 10 of layer1
    net.layer1.weight[0, 1] = 20 # change row 0 column 1 weight to 20 of layer1

# access all the parameters of the network
list_of_param = list(net.parameters())
```

## Cross-Entropy Loss
```python
import torch
import torch.nn as nn
import utils

mycrit = nn.CrossEntropyLoss()

# make a batch of labels, batch size = 2
# two classes: class 2 and class 3
labels = torch.LongTensor([2, 3])
# make a batch of scores, batch size = 2
# remember the net will auto transpose the output
# output scores is 2 x 3 matrix, one row = output scores for 1 input data
# 第一行的5对应的是标签2，是最大值；第二行的5对应的是标签3，是最大值
# 这是好的预测，因此损失值非常小
scores = torch.Tensor([[-1.2, 0.5, 5, -0.5], [1.4, -1.7, -1.3, 5.0]])
# 2 output scores for data size 2, 4 classes
utils.display_scores(scores)
average_loss = mycrit(scores, labels)
print('loss = ', average_loss.item()) # loss = 0.023508397862315178

# 第一行的3.1对应的是标签2，是最大值；第二行的2.0对应的是标签3，是最大值
# 这是比较好的预测，因为第二行的1.4的值也比较大，因此损失值增大
scores = torch.Tensor([[-1.2, 0.5, 3.1, -0.5], [1.4, -1.7 ,-1.3, 2.0]])
average_loss = mycrit(scores, labels)
print('loss = ', average_loss.item()) # loss = 0.2927485406398773

# 第一行的2.3对应的是标签1，是最大值；第二行的5.0对应的是标签2，是最大值
# 这是比较差的预测，因此损失值增大
scores = torch.Tensor([[0.8, 2.3, -1.0, -1.2], [-1.2, 1.3, 5.0, -2.0]])
average_loss = mycrit(scores, labels)
print('loss = ', average_loss.item()) # loss = 5.291047096252441

# 第一行的5对应的是标签2，是最大值；第二行的5对应的是标签3，是最大值
# 这是很好的预测，因此别的值都很小，因此损失值非常小，接近0
scores = torch.Tensor([[-5, -5, 5, -5], [-5, -5, -5, 5]])
average_loss = mycrit(scores, labels)
print('loss = ', average_loss.item()) # loss = 0.00013624693383462727
```

### 自行跟据公式计算loss
```python
import math
```

```python
# 两个输入样本(Batch Size = 2)和两个目标分类，一共4个分类
# 第一步
first = []
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    # labels[0] = 2
    # scores[0][2] = 5
    # labels[1] = 3
    # scores[1][3] = 5
    # 获得目标分类的分数
    val = scores[i][labels[i]]
    first.append(val)
print(first)

# 第二步
second = []
sum = 0
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    # 一共4个类别，因此range = 4
    for j in range(4):
        # softmax第一步：指数化
        # 把两个输入样本的4个类别的数据指数化并求和，求和是为了下面softmax第二步：归一化
        sum += math.exp(scores[i][j])
    second.append(sum)
    sum = 0
print(second)

L = []
# softmax第二步：归一化
# 分别求两个输入样本的loss
for i in range(2):
    # 这里是经过优化的公式，其实就是从对归一化的公式求log后得到的
    l = -first[i] + math.log(second[i])
    L.append(l)
print(L)

Loss = 0
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    Loss += L[i]
# 求平均损失
Loss = Loss / 2 # 需要除以2
print('自行计算的loss', Loss)
```

#### 使用LogSoftmax
```python
# 两个输入样本(Batch Size = 2)和两个目标分类，一共4个分类
# 第一步
first = []
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    # labels[0] = 2
    # scores[0][2] = 5
    # labels[1] = 3
    # scores[1][3] = 5
    # 获得目标分类的分数
    val = scores[i][labels[i]]
    first.append(val)
print(first)

# 直接求LogSoftmax
LogSoftmax = nn.LogSoftmax(dim=1)
log_probs = LogSoftmax(scores)
print("log_probs:\n", log_probs)

L = []
# 分别求两个输入样本的loss
for i in range(2):
    l = -log_probs[i][labels[i]]
    L.append(l)
print(L)

Loss = 0
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    Loss += L[i]
# 求平均损失
Loss = Loss / 2 # 需要除以2
print('自行计算的loss', Loss)
```

#### Softmax
```python
# 两个输入样本(Batch Size = 2)和两个目标分类，一共4个分类
# 第一步
first = []
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    # labels[0] = 2
    # scores[0][2] = 5
    # labels[1] = 3
    # scores[1][3] = 5
    # 获得目标分类的分数
    val = scores[i][labels[i]]
    first.append(val)
print(first)

# 直接求Softmax
Softmax = nn.Softmax(dim=1)
probs = Softmax(scores)
print("probs:\n", probs)

L = []
# 分别求两个输入样本的loss
for i in range(2):
    # 先求log
    l = -math.log(probs[i][labels[i]])
    L.append(l)
print(L)

Loss = 0
# 两个输入样本和两个目标分类，因此range = 2
for i in range(2):
    Loss += L[i]
# 求平均损失
Loss = Loss / 2 # 需要除以2
print('自行计算的loss', Loss)
```

## Multilayer Perceptron (MLP)
```python
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import time
import utils
```

### GPU or CPU
```python
# device = torch.device("cuda")
device = torch.device("cpu")
print(device)
```

### Download the data
```python
from utils import check_mnist_dataset_exists
data_path = check_mnist_dataset_exists()

# download mnist dataset
# 60,000 gray scale pictures as well as their label, each picture is 28 by 28 pixels
train_data = torch.load(data_path + 'mnist/train_data.pt')
train_label = torch.load(data_path + 'mnist/train_label.pt')
test_data = torch.load(data_path + 'mnist/test_data.pt')
test_label = torch.load(data_path + 'mnist/test_label.pt')
```

### Make a three layer net class
```python
class three_layer_net(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(three_layer_net, self).__init__()
        
        # 三层全连接网络MLP
        self.layer1 = nn.Linear(input_size, hidden_size1, bias=False)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2, bias=False)
        self.layer3 = nn.Linear(hidden_size2, output_size, bias=False)
        
    def forward(self, x):
        
        y = self.layer1(x)
        # 第一层使用ReLU作为激活函数
        y_hat = torch.relu(y)
        z = self.layer2(y_hat)
        # 第二层使用ReLU作为激活函数
        z_hat = torch.relu(z)
        # 第三层直接输出
        scores = self.layer3(z_hat)
        # prob = torch.softmax(y, dim=1)
        # 若这里用了softmax，则
        # 1. criterion需要用NLLLoss
        # 2. 需要对输出的分数求log，即log_scores = torch.log(scores)
        # 3. 最终的损失loss = criterion(log_scores, label)
        # 以上的步骤其实就是Cross-Entropy Loss的拆分，NLLLoss实际就是在做归一化
        # 若使用LogSoftmax
        # prob = torch.logsoftmax(y, dim=1)
        # 则不需要求log
        return scores
```

### Build the net
```python
# 784 = 28 x 28, 输出10个数字
net = three_layer_net(784, 50, 50, 10)
print(net)
utils.display_num_param(net)
```

### Send the weights of the networks to the GPU
```python
net = net.to(device)
```

### Choose the criterion and batch size
```python
# cross-entropy loss
criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
# batch size = 200
bs = 200
```

### Evaluate on test set
```python
def eval_on_test_set():

    running_error = 0
    num_batches = 0

    # test size = 10000
    for i in range(0, 10000, bs):

        # extract the minibatch
        minibatch_data = test_data[i: i + bs]
        minibatch_label = test_label[i: i + bs]

        # reshape the minibatch, 784 = 28 x 28
        # 200 x 784
        inputs = minibatch_data.view(bs, 784)

        # feed it to the network
        scores = net(inputs) 

        # compute the error made on this batch
        error = utils.get_error(scores, minibatch_label)

        # add it to the running error
        running_error += error.item()

        num_batches += 1

    total_error = running_error / num_batches
    print('test error  = ', total_error * 100, ' percent')
```

### Training loop
```python
start = time.time()

lr = 0.05 # initial learning rate

for epoch in range(200):
    
    # learning rate strategy: divide the learning rate by 1.5 every 10 epochs
    if epoch % 10 == 0 and epoch > 10:
        lr = lr / 1.5
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.
    optimizer = torch.optim.SGD(net.parameters() , lr=lr)
        
    running_loss = 0
    running_error = 0
    num_batches = 0
    
    # 先随机排序
    # train size = 60000
    shuffled_indices = torch.randperm(60000)

    # train size = 60000
    for count in range(0, 60000, bs):
        
        # forward and backward pass
        # set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()
        
        # 随机抽取200条数据, batch size = 200
        indices = shuffled_indices[count: count + bs]
        minibatch_data = train_data[indices]
        minibatch_label = train_label[indices]

        # send to the GPU
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        # reshape the minibatch, batch size = 200, 784 = 28 x 28
        # 200 x 784
        inputs = minibatch_data.view(bs, 784)

        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net
        scores = net(inputs) 
        # log_scores = torch.log(scores)

        # compute the average of the losses of the data points in the minibatch
        # 一个batch的平均损失
        loss = criterion(scores, minibatch_label) 
        # loss = criterion(log_scores, minibatch_label)
        
        # backward pass to compute dL/dU, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        
        # compute some stats
        # 获得当前的loss
        running_loss += loss.detach().item()
               
        error = utils.get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        
        num_batches += 1
    
    # compute stats for the full training set
    # once the epoch is finished we divide the "running quantities" by the number of batches
    # 总Loss = 每个Batch的Loss累加 / Batch数量累加 = 所有Batch的Loss / Batch数
    # 若Batch Size = 1，则Batch数 = 数据集大小
    # 若Batch Size = 数据集大小，则Batch数 = 1
    total_loss = running_loss / num_batches
    # 总Error = 每个Batch的Error累加 / Error数量累加 = 所有Batch的Error / Batch数
    # 若Batch Size = ，则Batch数 = 数据集大小
    # 若Batch Size = 数据集大小，则Batch数 = 1
    total_error = running_error / num_batches
    # 训练一个batch的时间
    elapsed_time = time.time() - start
    
    # every 10 epoch we display the stats and compute the error rate on the test set  
    if epoch % 10 == 0 : 
        print(' ')
        print('epoch = ', epoch, ' time = ', elapsed_time, ' loss = ', total_loss, ' error = ', total_error * 100, ' percent lr = ', lr)
        eval_on_test_set()
```

```python
# choose a picture at random
idx = randint(0, 10000-1)
im = test_data[idx]

# diplay the picture
utils.show(im)

# feed it to the net and display the confidence scores
# send to device, and view as a batch of 1 
im = im.to(device)

# im.view(1, 784)而不是im.view(784)是因为net是根据有batch size存在而设计的
# 例如batch size = 200，即im.view(200, 784)，则input是[[data_1], [data_2], ..., [data_200]]
# 而im.view(784)的input是[data_1]，少了一个维度
# 1代表了batch size = 1，就是只有一张图
im = im.view(1, 784) # one 1 x 784 image, 784 = 28 x 28
scores = net(im) 

# dim=1是因为这里的输出是
# [[-7.2764, 8.4730, 2.6842, 1.6302, -3.8437, -1.9697, -0.5854, -0.0792, 2.0861, -0.5462]]
# 需要求里面的维度的softmax
probs = torch.softmax(scores, dim=1)
utils.show_prob_cifar(probs.cpu())
```

## Convolutional Layer
```python
import torch
import torch.nn as nn

# Inputs: 2 channels
# Output: 5 activation maps
# 输入和输出都只是定义了深度
# Filters: 3x3
# padding: 1x1
# 卷积层并没有固定输入和输出图片的大小，输入图片可以是任意大小，输出图片的大小是有输入图片决定的
mod = nn.Conv2d(2, 5, kernel_size=3, padding=1)
```

## Pooling Layer
```python
import torch
import torch.nn as nn

# Inputs: activation maps of size n x n
# Output: activation maps of size n/p x n/p
# 输入和输出都只是定义了深度
# p: pooling size: 2x2
mod = nn.MaxPool2d(2, 2)
```

## VGG Architecture
```python
import torch
import torch.nn as nn
import torch.optim as optim
from random import randint
import time
import utils
```

### GPU or CPU
```python
# device = torch.device("cuda")
device = torch.device("cpu")
print(device)
```

### Download the data
```python
from utils import check_mnist_dataset_exists
data_path = check_mnist_dataset_exists()

# download cifar dataset
# 50,000 pictures as well as their label, each picture is 32 by 32 pixels
train_data = torch.load(data_path + 'cifar/train_data.pt')
train_label = torch.load(data_path + 'cifar/train_label.pt')
test_data = torch.load(data_path + 'cifar/test_data.pt')
test_label = torch.load(data_path + 'cifar/test_label.pt')
```

### Standardisation
```python
mean = train_data.mean() # 求tensor中所有元素的平均值
std = train_data.std() # 求tensor中所有元素的标准差
```

### Make a LeNet5 convnet class
```python
class LeNet5_convnet(nn.Module):

    def __init__(self):

        super(LeNet5_convnet, self).__init__()
        # CL1: 28 x 28 --> 50 x 28 x 28
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (28 + 2 * 1 - 3) / 1 + 1 = 28
        self.conv1 = nn.Conv2d(1, 50, kernel_size=3, padding=1)
        # MP1: 50 x 28 x 28 --> 50 x 14 x 14
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (28 - 2) / 2 + 1 = 14
        self.pool1  = nn.MaxPool2d(2, 2)

        # CL2: 50 x 14 x 14 --> 100 x 14 x 14
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (14 + 2 * 1 - 3) / 1 + 1 = 14
        self.conv2 = nn.Conv2d(50, 100, kernel_size=3, padding=1)
        # MP2: 100 x 14 x 14 --> 100 x 7 x 7
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (14 - 2) / 2 + 1 = 7
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # LL1: 100 x 7 x 7 = 4900 --> 100 
        self.linear1 = nn.Linear(4900, 100)
        # LL2: 100 --> 10
        self.linear2 = nn.Linear(100, 10)

    def forward(self, x):

        # CL1: 28 x 28 --> 50 x 28 x 28 
        x = self.conv1(x)
        x = torch.relu(x)

        # MP1: 50 x 28 x 28 --> 50 x 14 x 14
        # Pooling没有ReLU
        x = self.pool1(x)
        
        # CL2: 50 x 14 x 14  --> 100 x 14 x 14
        x = self.conv2(x)
        x = torch.relu(x)
        
        # MP2: 100 x 14 x 14 --> 100 x 7 x 7
        # Pooling没有ReLU
        x = self.pool1(x)

        # LL1: 100 x 7 x 7 = 4900  --> 100 
        # 这里用-1的原因是因为下面的设置了batch size = 128，这里自动换算维度，实际上-1可以用128代替
        x = x.view(-1, 4900)
        x = self.linear1(x)
        x = torch.relu(x)

        # LL2: 4900 --> 10 
        x = self.linear2(x)

        return x
```

### Make a VGG convnet class
```python
class VGG_convnet(nn.Module):

    def __init__(self):

        super(VGG_convnet, self).__init__()

        # block 1: 3 x 32 x 32 --> 64 x 16 x 16
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (32 + 2 * 1 - 3) / 1 + 1 = 32
        self.conv1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (32 - 2) / 2 + 1 = 16
        self.pool1 = nn.MaxPool2d(2, 2)

        # block 2: 64 x 16 x 16 --> 128 x 8 x 8
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (16 + 2 * 1 - 3) / 1 + 1 = 8
        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (16 - 2) / 2 + 1 = 8
        self.pool2 = nn.MaxPool2d(2, 2)

        # block 3: 128 x 8 x 8 --> 256 x 4 x 4
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (8 + 2 * 1 - 3) / 1 + 1 = 8
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (8 - 2) / 2 + 1 = 4
        self.pool3 = nn.MaxPool2d(2, 2)
        
        #block 4: 256 x 4 x 4 --> 512 x 2 x 2
        # 输出尺寸 = (输入尺寸n + 2 * 填充p - 过滤器尺寸f) / 步长s + 1
        # (4 + 2 * 1 - 3) / 1 + 1 = 4
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        # 输入尺寸 = (输入尺寸n - 过滤器尺寸f) / 步长s + 1
        # (4 - 2) / 2 + 1 = 2
        self.pool4  = nn.MaxPool2d(2, 2)

        # linear layers: 512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        self.linear1 = nn.Linear(2048, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 10)

    def forward(self, x):

        # block 1: 3 x 32 x 32 --> 64 x 16 x 16
        x = self.conv1a(x)
        x = torch.relu(x)
        x = self.conv1b(x)
        x = torch.relu(x)
        # Pooling没有ReLU
        x = self.pool1(x)

        # block 2: 64 x 16 x 16 --> 128 x 8 x 8
        x = self.conv2a(x)
        x = torch.relu(x)
        x = self.conv2b(x)
        x = torch.relu(x)
        # Pooling没有ReLU
        x = self.pool2(x)

        # block 3: 128 x 8 x 8 --> 256 x 4 x 4
        x = self.conv3a(x)
        x = torch.relu(x)
        x = self.conv3b(x)
        x = torch.relu(x)
        # Pooling没有ReLU
        x = self.pool3(x)

        #block 4: 256 x 4 x 4 --> 512 x 2 x 2
        x = self.conv4a(x)
        x = torch.relu(x)
        # Pooling没有ReLU
        x = self.pool4(x)

        # linear layers: 512 x 2 x 2 --> 2048 --> 4096 --> 4096 --> 10
        # 这里用-1的原因是因为下面的设置了batch size = 128，这里自动换算维度，实际上-1可以用128代替
        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        
        return x
```

### Build the net
```python
net = VGG_convnet()
print(net)
utils.display_num_param(net)
```

### Send the weights of the networks to the GPU
```python
net = net.to(device)
mean = mean.to(device)
std = std.to(device)
```

### Choose the criterion and batch size
```python
# cross-entropyloss
criterion = nn.CrossEntropyLoss()
my_lr = 0.25
bs = 128
```

### Evaluate on test set
```python
def eval_on_test_set():

    running_error = 0
    num_batches = 0

    # test size = 10000
    for i in range(0, 10000, bs):

        # extract the minibatch
        minibatch_data = test_data[i: i + bs]
        minibatch_label = test_label[i: i + bs]

        # send to the GPU
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        # normalize the minibatch (this is the only difference compared to before!)
        inputs = (minibatch_data - mean) / std

        # feed it to the network
        scores = net(inputs) 

        # compute the error made on this batch
        error = utils.get_error(scores, minibatch_label)

        # add it to the running error
        running_error += error.item()

        num_batches += 1

    total_error = running_error / num_batches
    print('test error  = ', total_error * 100, ' percent')
```

### Training loop
```python
start = time.time()

for epoch in range(1, 20):
    
    # divide the learning rate by 2 at epoch 10, 14 and 18
    if epoch == 10 or epoch == 14 or epoch==18:
        my_lr = my_lr / 2
    
    # create a new optimizer at the beginning of each epoch: give the current learning rate.
    optimizer = torch.optim.SGD(net.parameters() , lr=lr)
        
    running_loss = 0
    running_error = 0
    num_batches = 0
    
    # 先随机排序
    # train size = 50000
    shuffled_indices = torch.randperm(50000)

    # train size = 50000
    for count in range(0, 50000, bs):
        
        # forward and backward pass
        # set dL/dU, dL/dV, dL/dW to be filled with zeros
        optimizer.zero_grad()
        
        # 随机抽取200条数据, batch size = 200
        indices = shuffled_indices[count: count + bs]
        minibatch_data = train_data[indices]
        minibatch_label = train_label[indices]

        # send to the GPU
        minibatch_data = minibatch_data.to(device)
        minibatch_label = minibatch_label.to(device)

        # normalize the minibatch (this is the only difference compared to before!)
        inputs = (minibatch_data - mean) / std

        # tell Pytorch to start tracking all operations that will be done on "inputs"
        inputs.requires_grad_()

        # forward the minibatch through the net
        scores = net(inputs) 
        # log_scores = torch.log(scores)

        # compute the average of the losses of the data points in the minibatch
        # 一个batch的平均损失
        loss = criterion(scores, minibatch_label) 
        # loss = criterion(log_scores, minibatch_label)
        
        # backward pass to compute dL/dU, dL/dV and dL/dW
        loss.backward()

        # do one step of stochastic gradient descent: U=U-lr(dL/dU), V=V-lr(dL/dU), ...
        optimizer.step()
        
        # compute some stats
        # 获得当前的loss
        running_loss += loss.detach().item()
               
        error = utils.get_error(scores.detach(), minibatch_label)
        running_error += error.item()
        
        num_batches += 1
    
    # compute stats for the full training set
    # once the epoch is finished we divide the "running quantities" by the number of batches
    # 总Loss = 每个Batch的Loss累加 / Batch数量累加 = 所有Batch的Loss / Batch数
    # 若Batch Size = 1，则Batch数 = 数据集大小
    # 若Batch Size = 数据集大小，则Batch数 = 1
    total_loss = running_loss / num_batches
    # 总Error = 每个Batch的Error累加 / Error数量累加 = 所有Batch的Error / Batch数
    # 若Batch Size = ，则Batch数 = 数据集大小
    # 若Batch Size = 数据集大小，则Batch数 = 1
    total_error = running_error / num_batches
    # 训练一个batch的时间
    elapsed_time = time.time() - start
    
    # every 10 epoch we display the stats and compute the error rate on the test set  
    if epoch % 10 == 0 : 
        print(' ')
        print('epoch = ', epoch, ' time = ', elapsed_time, ' loss = ', total_loss, ' error = ', total_error * 100, ' percent lr = ', lr)
        eval_on_test_set()
```

```python
# choose a picture at random
idx = randint(0, 10000-1)
im = test_data[idx]

# diplay the picture
utils.show(im)

# feed it to the net and display the confidence scores
# send to device, rescale, and view as a batch of 1 
im = im.to(device)
im = (im - mean) / std

# im.view(1, 3, 32, 32)而不是im.view(3, 32, 32)是因为net是根据有batch size存在而设计的
# 例如batch size = 128，即im.view(128, 784)，则input是[[data_1], [data_2], ..., [data_128]]
# 而im.view(3, 32, 32)的input是[data_1]，少了一个维度
# 1代表了batch size = 1，就是只有一张图
im = im.view(1, 3, 32, 32) # one 1 x (3 x 32 x 32) image, 3072 = 3 x 32 x 32
scores = net(im) 

# dim=1是因为这里的输出是
# [[-7.2764, 8.4730, 2.6842, 1.6302, -3.8437, -1.9697, -0.5854, -0.0792, 2.0861, -0.5462]]
# 需要求里面的维度的softmax
probs = torch.softmax(scores, dim=1)
utils.show_prob_cifar(probs.cpu())
```
