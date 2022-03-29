'''
https://www.acmicpc.net/problem/2292
벌집 위치찾기
'''

inputs = int(input())
input_ = inputs - 1
count = 1
for i in range(1, inputs):
    input_ = input_ - (6 * i)
    count += 1
    if input_ <= 0:
        break
    
print(count)