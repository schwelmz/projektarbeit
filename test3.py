N = 100
while N < 2000:
    print(N)
    N = N*2-1

print(" ")

for mult in range(1,10):
    N=100
    for i in range(1,mult):
        N = N*2-1
    print(mult, " ",N)