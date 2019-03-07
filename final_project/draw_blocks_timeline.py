import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

filename = "thread_log/opt.txt"

sm_num = 15
lines_per_kernel = 6 * sm_num + 1
mod_num = 7
shared_mem_per_block = 4    # KB

if shared_mem_per_block == 0:
    max_support_kernel_id = 31
else:
    max_support_kernel_id = min(96 / shared_mem_per_block - 1, 31)

start_time_list = []
stop_time_list = []
kernel_id_list = []
block_id_list = []

sm_line_list = []
    
for i in range (sm_num):
    start_time_list.append([])
    stop_time_list.append([])
    kernel_id_list.append([])
    block_id_list.append([])
    sm_line_list.append([])

with open(filename) as f:
    for kernel_line_num, line in enumerate(f):
        if 'Kernel number' in line:
            a, b = line.split(":")
            kernel_num = int(b)
            
with open(filename) as f:
    for sm_line_num, line in enumerate(f):
        if 'SM id: 0' in line:
            sm_line_list[0].append(sm_line_num)
        if 'SM id: 1' in line and 'SM id: 10' not in line and 'SM id: 11' not in line and 'SM id: 12' not in line and 'SM id: 13' not in line and 'SM id: 14' not in line:
            sm_line_list[1].append(sm_line_num)
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

# for row in range(sm_num):
#     cols = len(sm_line_list[row])
#     print("\n Row", row, "has", cols, "columns: ", end="")
#     for col in range(cols):
#         print(sm_line_list[row][col], " ", end="")

for i in range(sm_num - 1):
    idx = 0
    with open(filename) as f:
        for ln, tl in enumerate(f):
            if idx >= len(sm_line_list[i]):
                break
            if ln == sm_line_list[i][idx] - 2:  # block id
                a, b = tl.split(":")
                b = int(b)
                block_id_list[i].append(b)
            if ln == sm_line_list[i][idx] - 1:  # kernel id
                a, b = tl.split(":")
                b = int(b)
                kernel_id_list[i].append(b)
            if ln == sm_line_list[i][idx] + 1:   # start_time @smid = 0
                a, b = tl.split(":")
                b = int(b)
                start_time_list[i].append(b)
            if ln == sm_line_list[i][idx] + 2:   # stop_time @smid = 0
                a, b = tl.split(":")
                b = int(b)
                stop_time_list[i].append(b)
                idx += 1

# now we get the start/stop time of each block @smid = 0
# plot
height = max_support_kernel_id + 1
u_shift = 0.3
# plt.figure()
for i in range (sm_num-1):
# for i in range (13, 14):
    plt.figure()
    plt.title('smid' + str(i))
    # plt.subplot(3, 1, i+1)
    cnt_0 = 0
    cnt_1 = 0 
    cnt_2 = 0
    cnt_3 = 0
    cnt_4 = 0
    cnt_5 = 0
    cnt_6 = 0
    for j in range (len(start_time_list[i])):
        y_h = kernel_num - kernel_id_list[i][j]
        # y_h = height
        xmin_h = start_time_list[i][j]
        xmax_h = stop_time_list[i][j]
        x_t = (stop_time_list[i][j]+start_time_list[i][j])/2 
        y_t = y_h + 0.05 # 0.25
        s_t = 'K'+str(kernel_id_list[i][j])+': '+str(block_id_list[i][j])
        if kernel_id_list[i][j] > max_support_kernel_id:
            # m = int(kernel_id_list[i][j] / max_support_kernel_id)
            y_h = (kernel_num - kernel_id_list[i][j]) + max_support_kernel_id + 2# + 1 - 0.3
            y_t = y_h + 0.1
        if kernel_id_list[i][j] % mod_num == 0:
            offset = u_shift * cnt_0
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='r', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_0 += 1
            # plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(formatter))
            # plt.barh(range(len(start_time_list[i])), int(stop_time_list[i][j]) - int(start_time_list[i][j]), left=int(start_time_list[i][j]), color='r')
            # plt.yticks(range(len(start_time_list[i])), str(kernel_id_list[i][j]))
        elif int(kernel_id_list[i][j]) % mod_num == 1:
            offset = u_shift * cnt_1
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='y', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_1 += 1
        elif int(kernel_id_list[i][j]) % mod_num == 2:
            offset = u_shift * cnt_2
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='g', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_2 += 1
        elif int(kernel_id_list[i][j]) % mod_num == 3:
            offset = u_shift * cnt_3
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='b', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_3 += 1
        elif int(kernel_id_list[i][j]) % mod_num == 4:
            offset = u_shift * cnt_4
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='c', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_4 += 1
        elif int(kernel_id_list[i][j]) % mod_num == 5:
            offset = u_shift * cnt_5
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='m', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_5 += 1
        elif int(kernel_id_list[i][j]) % mod_num == 6:
            offset = u_shift * cnt_6
            plt.hlines(y=y_h-offset, xmin=xmin_h, xmax=xmax_h, colors='k', lw=6)
            plt.text(x=x_t, y=y_t-offset, s=s_t, horizontalalignment='center')
            plt.text(x=start_time_list[i][j], y=y_t-offset, s=str(start_time_list[i][j]), horizontalalignment='left')
            plt.text(x=stop_time_list[i][j], y=y_t-offset, s=str(stop_time_list[i][j]), horizontalalignment='right')
            cnt_6 += 1

        # height -= 1
        # if height == 0:
        #     height = max_support_kernel_id + 1

plt.show()