import json

from tqdm import tqdm

labels = []
with open("/home/yuhaiyang/BackdoorMBTI/data/hmdb51/train_labels.json", "r") as f:
    labels = json.load(f)

train_indices = []
added_labels = []
cnt = 0
num_samples = 20
for cur_class in range(51):
    for idx in tqdm(range(len(labels))):
        if cur_class == labels[idx]:
            train_indices.append(idx)
            added_labels.append(labels[idx])
            cnt += 1
            if cnt >= num_samples:
                cnt=0
                break

with open("/home/yuhaiyang/BackdoorMBTI/data/hmdb51/test_labels.json", "r") as f:
    labels = json.load(f)

test_indices = []
added_labels = []
cnt = 0

for cur_class in range(51):
    for idx in tqdm(range(len(labels))):
        if cur_class == labels[idx]:
            test_indices.append(idx)
            added_labels.append(labels[idx])
            cnt += 1
            if cnt >= num_samples:
                cnt = 0
                break

print("length of train_indices: ", len(train_indices))
print("length of test_indices: ", len(test_indices))
print(added_labels)
with open("train_indices.json", "w") as f:
    json.dump(train_indices, f)

with open("test_indices.json", "w") as f:
    json.dump(test_indices, f)
