import numpy as np

detect_and_num={}
label = []
r_label = []

line_list=[
    {'detect':'windows.virus'},
    {'detect': 'ubuntu.virus'},
    {'detect': 'windows.virus'},
    {'detect': 'linux.virus'},
]





for line_a in line_list:
    detect_val = line_a['detect'].split(".")[0]

    if len(detect_and_num) == 0:
        detect_and_num[detect_val] = 1

    if detect_val not in detect_and_num.keys():
        current_max = detect_and_num[max(detect_and_num, key=detect_and_num.get)]
        detect_and_num[detect_val] = current_max + 1
    # 라벨은 숫자로 표현
    label.append(detect_and_num[detect_val])

max_num = max(label)
for l_num in label:
    lst = [0 for _ in range(max_num)]
    lst[-l_num] = 1
    r_label.append(lst)

r_label = np.array(r_label)
print(r_label)
print(detect_and_num)

np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(r_label)))
print(r_label[shuffle_indices])