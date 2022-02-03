# squeeze和unsqueeze方法测试

import torch
# torch.squeeze() 把维度为1的维去掉,(和i能去掉维度为1的, 没有维度为1的维, 则保持原样)
a = torch.rand(2,3,4)
b = torch.unsqueeze(a, 2)
print("a =", a, a.size())
print("b =", b, b.size())
c = torch.squeeze(b)
print("c =", c, c.size())
d = torch.squeeze(a)
print("d =", d, d.size())

# torch.expand()这个函数的作用就是对指定的维度进行数值大小的改变。只能改变维大小为1的维，否则就会报错。不改变的维可以传入-1或者原来的数值。
a = torch.rand(1,2,3)
print("a =", a, a.size())
d = a.expand(3, -1, -1)
print("d =", d, d.size())

# torch.repeat()沿着指定的维度，对原来的tensor进行数据复制。这个函数和expand()还是有点区别的。expand()只能对维度为1的维进行扩大，
# 而repeat()对所有的维度可以随意操作。
a = torch.rand(1,2,3)
print("a =", a, a.size())
d = a.repeat(3, 3, 3)
print("d =", d, d.size())

from tqdm import tqdm
import time
for i in tqdm(range(100)):
    time.sleep(0.1)