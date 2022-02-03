# Pytorch搭建简单神经网络（一）——回归
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt

#随机生成一组数据,并加上随机噪声增加数据的复杂性
x = torch.linspace(-1, 1, 100)
# print("x=", x)
x = torch.unsqueeze(x, dim=1)
# print("x=", x)
y = x.pow(3)+0.1*torch.randn(x.size())
# print("y=", y)


# 将数据转化成Variable的类型用于输入神经网络,为了更好的看出生成的数据类型，我们采用将生成的数据plot出来
# 目前感觉这句话没啥用，Variable之前和之后还都是tensor，类型，维度，都没有任何变化。之前能画图，之后也能画图
x , y =(Variable(x),Variable(y))
# print(x, y)

# 开始搭建神经网络，以下作为搭建网络的模板，定义类，然后继承nn.Module，再继承自己的超类。
class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)
        # self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):
        out = self.hidden1(input)
        out = torch.relu(out)
        # out = self.hidden2(out)
        # out = torch.sigmoid(out)
        out =self.predict(out)

        return out

myNet = Net(1,10,1)
# print(myNet)

"""
构建优化目标及损失函数
torch.optim是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性。为了使用torch.optim，
你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。
为了构建一个Optimizer，你需要给它一个包含了需要优化的参数（必须都是Variable对象）的iterable。
然后，你可以设置optimizer的参 数选项，比如学习率，权重衰减，等等。[1]
"""

optimizer = torch.optim.SGD(myNet.parameters(),lr = 0.1)
loss_func = torch.nn.MSELoss()

"""
采用随机梯度下降进行训练，损失函数采用常用的均方损失函数，设置学习率为0.1，可以根据需要进行设置，原则上越小学习越慢，
但是精度也越高，然后进行迭代训练（这里设置为5000次）
"""
# print("parameter=",myNet.hidden1.parameters())
for parameter in myNet.parameters():
    print("parameter=",parameter.data.numpy())

for t in tqdm(range(5000)):
    prediction = myNet(x)
    loss = loss_func(prediction,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

for parameter in myNet.parameters():
    print("parameter=",parameter.detach().numpy())

predictX = torch.linspace(x.min(),x.max(),100)
predictY = myNet(x)
# print("predictX=", predictX)
# print("predictY=", predictY)

plt.scatter(x, y, c='b', marker='.')
plt.plot(torch.unsqueeze(predictX, dim=1).detach().numpy(), predictY.detach().numpy(), 'r-')
plt.show()
