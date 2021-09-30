'''
https://www.acmicpc.net/problem/1193
분수 찾기
'''
inputs = int(input())
line = 1

while inputs > line:
    inputs -= line
    line += 1

if line % 2 == 0:
    a = inputs
    b = line - inputs + 1
elif line % 2 == 1:
    a = line - inputs + 1
    b = inputs

print(str(a)+'/'+str(b))