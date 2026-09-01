import torch

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s)


"""
a.
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
so it worked best all in fp32, second is accumulater s in fp32, and delta in fp16, it doesnt matter if delta cast to fp32 before accumulate to s.

so the key point is s need fp32, that means  attn can in bf16, weights,  grad, opt states need accumulate in fp32.


"""



"""
b.
model parameter: fp32

output of ffn: fp16 (hardware do fp32 MAC accumulate internally)
output of layer norm: fp16 (mean and variance reductions are sensitve)

perdicted logits: fp16
loss: fp32
grads: fp32


"""

"""
c.
fp16: e5m10, bf16: e8m7, fp32: e8m23
bf16 is lower precision but higher range than fp16, same range as fp32 (8 exponent bits), so its safe for layer norm 

"""