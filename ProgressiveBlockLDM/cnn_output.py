import math
f=32
k=4
s=2
p=1
n=4

for i in range(n):
    o = math.floor(((f-k+2*p)/s)+1)
    print(o)
    f = o
