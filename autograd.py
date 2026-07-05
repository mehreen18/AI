import torch
x=torch.tensor(5.0, requires_grad=True)
##y=x**2
a=x+2
y=x**a
y.backward()
print(x.grad)


