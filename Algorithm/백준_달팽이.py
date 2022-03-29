'''
https://www.acmicpc.net/problem/2869
달팽이
'''
import math

up, down, height = list(map(int, input().split()))
total = (height - down) / (up - down)

print(math.ceil(total))