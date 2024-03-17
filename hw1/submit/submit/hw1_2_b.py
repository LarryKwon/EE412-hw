import re
import sys
import time

file_name = sys.argv[1]
all_items = {}
all_baskets = []


def find_occurance(item1: str, item2: str):
    count = 0
    for basket in all_baskets:
        if item1 in basket and item2 in basket:
            count += 1
    return count


start_time = time.time()
with open(file_name, 'r') as file:
    # Loop through each line in the file
    for line in file:
        items = [item for item in re.split(r'[^\w]+', line) if item]
        basket_set = set()
        for item in items:
            basket_set.add(item)
            if item not in all_items:
                all_items[item] = 1
            else:
                all_items[item] += 1
        all_baskets.append(basket_set)

all_items = {k: v for k, v in sorted(
    all_items.items(), key=lambda item: item[1])}

not_frequent_items = []
for key in all_items:
    if all_items[key] < 100:
        not_frequent_items.append(key)

for item in not_frequent_items:
    del all_items[item]

end_of_frequent_item = time.time()
# print(f'{len(all_items.keys())}, time: {end_of_frequent_item - start_time}')
print(f'{len(all_items.keys())}')


singleton_size = len(all_items.keys())
singletons = list(all_items.keys())
pairs = [[None] * singleton_size] * singleton_size

pairs_size = 0
frequent_pairs = {}
for i in range(singleton_size):
    for j in range(i+1, singleton_size):
        if pairs[i][j] == None:
            pairs[i][j] = 0
        # if (j == singleton_size - 1):
        count = find_occurance(singletons[i], singletons[j])
        if count >= 100:
            pairs[i][j] = count
            key = (singletons[i], singletons[j])
            frequent_pairs[key] = count
            pairs_size += 1
    end_of_each_pairs = time.time()
    # print(f'i: {i}, total: {singleton_size}, time: {end_of_each_pairs - end_of_frequent_item}')

frequent_pairs = {k: v for k, v in sorted(
    frequent_pairs.items(), key=lambda item: -item[1])}

top10_frequent_pairs = []
index = 0
for pair in frequent_pairs:
    if (index >= 10):
        break
    top10_frequent_pairs.append((pair, frequent_pairs[pair]))
    index += 1
end_of_frequent_pairs = time.time()
# print(f'{pairs_size}, time: {end_of_each_pairs - end_of_frequent_item}')
print(f'{pairs_size}')

for pair in top10_frequent_pairs:
    item1 = pair[0][0]
    item2 = pair[0][1]
    count = pair[1]
    conf12 = count / all_items[item1]
    conf21 = count / all_items[item2]
    print(f'{item1}\t{item2}\t{count}\t{conf12:.4f}\t{conf21:.4f}')
