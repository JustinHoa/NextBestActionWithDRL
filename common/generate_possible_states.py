# Số prefix có thể có:
# Gender = 1 (Male), Marital Status = 0/1 
    # (6 + (2**3 - 1) + (2**10 - 1)) * 2 = 2072
# Gender = 0, Marital Status = 0
    # 6 + (2**4 - 1) + (2**10 - 1) = 1044
# Gender = 0, Marital Status = 1
    # 6 + (2**5 - 1) + (2**10 - 1) = 1060
# -> Tổng: 4176
# Bỏ đi goal state.

from itertools import product
import pandas as pd
import os
import itertools

output_dir = os.path.join('data/raw/')

all_prefixs = [] 

for i in range(6):
    prefix = [1] * i + [0] * (5 - i)
    full_prefix = prefix + [0] * (21 - len(prefix))
    all_prefixs.append(full_prefix)

clinical_generator = itertools.product([0,1], repeat=5)
clinical_prefix = [list(combo) for combo in clinical_generator if any(combo)] # Bỏ trường hợp [0] * 5

for i in range(len(clinical_prefix)):
    prefix = [1] * 5 + clinical_prefix[i] + [0] * 11
    all_prefixs.append(prefix)

paraclinical_generator = itertools.product([0,1], repeat=10)
paraclinical_prefix = [list(combo) for combo in paraclinical_generator if any(combo)] # Bỏ trường hợp [0] * 10

for i in range(len(paraclinical_prefix)):
    prefix = [1] * 10 + paraclinical_prefix[i] + [0]
    all_prefixs.append(prefix)

# all_prefixs.append([1] * 21)

# Tạo prefix cho male
all_prefix_for_male = []
temp_set = set()

for i in range(len(all_prefixs)):
    prefix = all_prefixs[i].copy()
    prefix[8] = 0
    prefix[9] = 0
    temp_set.add(tuple(prefix))

all_prefix_for_male = [list(p) for p in temp_set] 

# Tạo prefix cho Female, Not Married
all_prefix_for_female_not_married = []
temp_set = set()

for i in range(len(all_prefixs)):
    prefix = all_prefixs[i].copy()
    prefix[8] = 0
    temp_set.add(tuple(prefix))

all_prefix_for_female_not_married = [list(p) for p in temp_set] 

# Lưu file
df = []
for i in range(len(all_prefixs)):
    df.append([0, 1] + all_prefixs[i])
for i in range(len(all_prefix_for_male)):
    df.append([1, 0] + all_prefix_for_male[i])
    df.append([1, 1] + all_prefix_for_male[i])
for i in range(len(all_prefix_for_female_not_married)):
    df.append([0, 0] + all_prefix_for_female_not_married[i])

print(f"✅ Found {len(df)} states.")

file_path = os.path.join(output_dir, "possible_states.csv")
df = pd.DataFrame(df)
df.to_csv(file_path, index=False, header=False)

print(f"✅ Saved possible states successfully at {file_path}")

