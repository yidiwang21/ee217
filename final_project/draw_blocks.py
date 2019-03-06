import matplotlib.pyplot as plt
import numpy as np
from pylab import *

sm_num = 15
kernel_num  = 33
lines_per_kernel = 6 * sm_num + 1

filename = "log_7.txt"

start_time_list = []
stop_time_list = []

sm_line_list = []

for i in range (sm_num):
    start_time_list.append([])
    stop_time_list.append([])
    sm_line_list.append([])

with open(filename) as f:
    for sm_line_num, line in enumerate(f):
        if 'SM id: 0' in line:
            sm_line_list[0].append(sm_line_num)
        # if 'SM id: 1' in line:
        #     sm_line_list[1].append(sm_line_num)
        # TODO: will mess up with 1x
        if 'SM id: 2' in line:
            sm_line_list[2].append(sm_line_num)
        if 'SM id: 3' in line:
            sm_line_list[3].append(sm_line_num)
        if 'SM id: 4' in line:
            sm_line_list[4].append(sm_line_num)
        if 'SM id: 5' in line:
            sm_line_list[5].append(sm_line_num)
        if 'SM id: 6' in line:
            sm_line_list[6].append(sm_line_num)
        if 'SM id: 7' in line:
            sm_line_list[7].append(sm_line_num)
        if 'SM id: 8' in line:
            sm_line_list[8].append(sm_line_num)
        if 'SM id: 9' in line:
            sm_line_list[9].append(sm_line_num)
        if 'SM id: 10' in line:
            sm_line_list[10].append(sm_line_num)
        if 'SM id: 11' in line:
            sm_line_list[11].append(sm_line_num)
        if 'SM id: 12' in line:
            sm_line_list[12].append(sm_line_num)
        if 'SM id: 13' in line:
            sm_line_list[13].append(sm_line_num)
        if 'SM id: 14' in line:
            sm_line_list[14].append(sm_line_num)

for row in range(sm_num):
    cols = len(sm_line_list[row])
    print("\n Row", row, "has", cols, "columns: ", end="")
    for col in range(cols):
        print(sm_line_list[row][col], " ", end="")

for i in range(sm_num - 1):
    idx = 0
    with open(filename) as f:
        for ln, tl in enumerate(f):
            if idx >= len(sm_line_list[i]):
                break;
            if ln == sm_line_list[i][idx] + 1:   # start_time line @smid = 0
                a, b = tl.split(":")
                b = int(b)
                start_time_list[i].append(b)
            if ln == sm_line_list[i][idx] + 2:   # stop_time line @smid = 0
                a, b = tl.split(":")
                b = int(b)
                stop_time_list[i].append(b)
                idx += 1

# now we get the start/stop time of each block @smid = 0
# plot
height = 1
# fig, ax = plt.subplots(nrows=15, ncols=1)
plt.figure()
for i in range (sm_num - 1):
    plt.subplot(sm_num - 1, 1, i+1)
    for j in range (len(start_time_list[i])):
        plt.hlines(y=height, xmin=int(start_time_list[i][j]), xmax=int(stop_time_list[i][j]), lw=2)
        height += 1;

plt.show()