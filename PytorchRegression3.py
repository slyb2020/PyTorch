import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(3)+0.1*torch.randn(x.size())

x , y =(Variable(x),Variable(y))

# class Net(nn.Module):
#     def __init__(self,n_input,n_hidden,n_output):
#         super(Net,self).__init__()
#         self.hidden1 = nn.Linear(n_input,n_hidden)
#         self.hidden2 = nn.Linear(n_hidden,n_hidden)
#         self.predict = nn.Linear(n_hidden,n_output)
#     def forward(self,input):
#         out = self.hidden1(input)
#         out = F.relu(out)
#         out = self.hidden2(out)
#         out = F.relu(out)
#         out =self.predict(out)
#
#         return out
# net = torch.nn.Sequential(
#     nn.Linear(1,20),
#     torch.nn.ReLU(),
#     nn.Linear(20,20),
#     torch.nn.ReLU(),
#     nn.Linear(20,1)
# )
#
# # net = Net(1,20,1)
# print(net)
# optimizer = torch.optim.SGD(net.parameters(),lr = 0.05)
# loss_func = torch.nn.MSELoss()
#
# plt.ion()
# plt.show()
#
# for t in range(2000):
#     prediction = net(x)
#     loss = loss_func(prediction,y)
#
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     if t%200 ==0:
#         plt.cla()
#         plt.scatter(x.data.numpy(), y.data.numpy())
#         plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
#         plt.text(0.5, 0, 'Loss = %.4f' % loss.data, fontdict={'size': 20, 'color': 'red'})
#         plt.pause(0.05)
#
# torch.save(net,'net.pkl')
# torch.save(net.state_dict(),'net_parameter.pkl')
#
net1 = torch.load('net.pkl')
net2 = torch.nn.Sequential(
    nn.Linear(1,20),
    torch.nn.ReLU(),
    nn.Linear(20,20),
    torch.nn.ReLU(),
    nn.Linear(20,1)
)
net2.load_state_dict(torch.load('net_parameter.pkl'))

prediction1 = net1(x)
prediction2 = net2(x)

plt.figure(1,figsize=(10,3))
plt.subplot(121)
plt.title('net1')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction1.data.numpy(), 'r-', lw=5)
# plt.show()

plt.figure(1,figsize=(10,3))
plt.subplot(122)
plt.title('net2')
plt.scatter(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), prediction2.data.numpy(), 'r-', lw=5)
plt.show()