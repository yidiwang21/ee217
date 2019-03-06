import matplotlib.pyplot as plt
import numpy as np

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
    for sm_line_num, line in enumerate(f, 1):
        if 'SM id: 0' in line:
            sm_line_list[0].append(sm_line_num)
        # TODO:

for row in range(sm_num):
    cols = len(sm_line_list[row])
    print("Row", row, "has", cols, "columns: ", end="")
    for col in range(cols):
        print(sm_line_list[row][col], " ", end="")

idx = 0
with open(filename) as f:
    for line_num, line in enumerate(f, 1):
        if line_num == sm_line_list[0][idx] + 1:   # start_time line @smid = 0
            a, b = line.split(":")
            b = int(b)
            start_time_list[0].append(b)
            idx += 1
        if line_num == sm_line_list[0][idx] + 2:   # stop_time line @smid = 0
            a, b = line.split(":")
            b = int(b)
            stop_time_list[0].append(b)
            idx += 1

# now we get the start/stop time of each block @smid = 0
# plot
height = 1
fig, ax = plt.subplots(nrows=15, ncols=1)
for i in range (0, 14):
    for j in range (0, len(start_time_list[i])):
        ax.plot(start_time_list[i][j], stop_time_list[i][j])




