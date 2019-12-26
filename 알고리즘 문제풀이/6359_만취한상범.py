T = int(input())
for i in range(T):
    n = int(input())
    prison = [1 for i in range(n+1)]
    for j in range(1,n+1):
        for k in range(j,n+1):
            if k % j == 0 :
                prison[k] = -(prison[k])
    count = 0
    for j in range(1,n+1):
        if prison[j] == -1 :
            count += 1
    print(count)