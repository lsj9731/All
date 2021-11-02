import random

list_file = 'C:/Users/82107/Downloads/filelist/filelist.txt'

def get_data(data):
    return data[0], int(data[1]), int(data[2])

data = [get_data(x.split(' ')) for x in open(list_file)]

sum_wake = [] 
sum_bg = []

print('Get filelist labels')

[sum_wake.append(labels) for labels in data if labels[-1] == 1]
[sum_bg.append(labels) for labels in data if labels[-1] == 0]

print('Done.\n')

# Prints list of random items of given length
sample_bg = random.sample(sum_bg, len(sum_wake))

data_list = sum_wake + sample_bg
random.shuffle(data_list)

f = open("C:/Users/82107/Downloads/filelist/filelist_1025.txt", 'w')

for datas in data_list:
    path = datas[0]
    frame = str(datas[1])
    label = str(datas[2])
    
    f.write(str(path)+' ')
    f.write(str(frame)+' ')
    f.write(str(label)+'\n')
    
f.close()