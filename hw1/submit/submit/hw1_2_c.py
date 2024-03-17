import re
import sys
import time

file_name = sys.argv[1]
THRESHOLD=100

def find_occurance(item1: str, item2: str):
    count = 0
    for basket in all_baskets:
        if item1 in basket and item2 in basket:
            count += 1
    return count


def find_occurance_triple(item1: str, item2: str, item3: str):
    count = 0
    for basket in all_baskets:
        if item1 in basket and item2 in basket and item3 in basket:
            count += 1
    return count


def make_triples(pair_set):
    triples = {}

    # Convert the pair set into a set of frozensets for efficient membership testing
    pair_set = {frozenset(pair) for pair in pair_set}

    triple_set = set()
    for pair in pair_set:
        for element in pair:
            # Find all pairs containing the current element
            related_pairs = {frozenset(p) for p in pair_set if element in p}

            # Check if all such related pairs are in the original pair_set
            if related_pairs.issubset(pair_set):
                # If so, add the elements of the current pair to a potential triple
                for related_pair in related_pairs:
                    triple = set(pair.copy())
                    remain = (triple | related_pair) - (triple & related_pair)
                    # print(triple, related_pair, remain)
                    if (remain in pair_set):
                        triple.update(related_pair)
                        if (len(triple) >= 3):
                            # Add the potential triple to the set of triples
                            triple_set.add(frozenset(triple))

    for triple in list(triple_set):
        triples[triple] = 0

    return triples


def init():
    all_items = {}
    all_baskets = []
    with open(file_name, 'r') as file:
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
    # sorting
    all_items = {k: v for k, v in sorted(
        all_items.items(), key=lambda item: item[1])}
    # print(len(all_items))
    return all_items, all_baskets


def make_frequent_items(all_items: dict):
    not_frequent_items = []
    copy_all_items = all_items.copy()
    for key in copy_all_items:
        if copy_all_items[key] < THRESHOLD:
            not_frequent_items.append(key)

    for item in not_frequent_items:
        del copy_all_items[item]

    return copy_all_items


def make_pairs(singletons):
    pairs = {}
    pairs_size = 0
    frequent_pairs = {}
    index = 0
    start_time = time.time()
    size = len(singletons)
    for i in range(len(singletons)):
        for j in range(i + 1, len(singletons)):
            # pairs[pair] = 0
            count = find_occurance(singletons[i], singletons[j])
            if count >= THRESHOLD:
                pair = frozenset([singletons[i],singletons[j]])
                frequent_pairs[pair] = count
                pairs_size += 1
        end_time = time.time()
        # print(f'{i} / {size}, time: {end_time - start_time}')
    return frequent_pairs, pairs_size


def make_frequent_pairs(pairs: dict):
    pairs_size = 0
    frequent_pairs = {}
    index = 0
    size = len(pairs)
    for pair in pairs:
        pair_copy = list(pair)
        item1 = pair_copy[0]
        item2 = pair_copy[1]
        count = find_occurance(item1, item2)
        if count >= THRESHOLD:
            frequent_pairs[pair] = count
            pairs_size += 1
        # if(index % 10000 == 0):
        #     print(f'{index} / {size}')
        # index+=1
    return (frequent_pairs, pairs_size)


def make_frequent_triples(triples: dict):
    triples_size = 0
    frequent_triples = {}
    for triple_set in triples:
        triple = tuple(triple_set)
        item1 = triple[0]
        item2 = triple[1]
        item3 = triple[2]
        count = find_occurance_triple(item1, item2, item3)
        if count >= THRESHOLD:
            frequent_triples[triple] = count
            triples_size += 1
    return frequent_triples, triples_size


def top10(frequent: dict):
    frequent = {k: v for k, v in sorted(
        frequent.items(), key=lambda item: -item[1])}
    top10_frequent_pairs = []
    index = 0
    for pair in frequent:
        if (index >= 10):
            break
        top10_frequent_pairs.append((pair, frequent[pair]))
        index += 1
    return top10_frequent_pairs


start_time = time.time()
all_items, all_baskets = init()
frequent_items = make_frequent_items(all_items)
end_of_frequent_item = time.time()
# print(f'{len(frequent_items.keys())}')


singleton_size = len(frequent_items.keys())
singletons = list(frequent_items.keys())

pairs, pairs_size = make_pairs(singletons)
end_of_pairs = time.time()
# print(f'{len(pairs)}, time: {end_of_pairs - end_of_frequent_item}')


# frequent_pairs, pairs_size = make_frequent_pairs(pairs)

frequent_pairs = pairs

frequent_pairs = {k: v for k, v in sorted(
    frequent_pairs.items(), key=lambda item: -item[1])}

end_of_frequent_pairs = time.time()
# print(f'{pairs_size}, time: {end_of_frequent_pairs - end_of_frequent_item}')
# print(f'{pairs_size}')


triples = make_triples(frequent_pairs.keys())
frequent_triples, triples_size = make_frequent_triples(triples)
end_of_frequent_triples = time.time()
# print(f'{triples_size}, time: {end_of_frequent_triples - end_of_frequent_pairs}')
print(triples_size)

top10_frequent = top10(frequent_triples)

frequent_triples = {k: v for k, v in sorted(
    frequent_triples.items(), key=lambda item: -item[1])}

for triple in top10_frequent:
    item1 = triple[0][0]
    item2 = triple[0][1]
    item3 = triple[0][2]
    count = triple[1]

    sort_items = []
    sort_items.append(item1)
    sort_items.append(item2)
    sort_items.append(item3)
    sort_items.sort()
    item1 = sort_items[0]
    item2 = sort_items[1]
    item3 = sort_items[2]
    item12 = set()
    item12.add(sort_items[0])
    item12.add(sort_items[1])
    item12 = frozenset(item12)

    item13 = set()
    item13.add(sort_items[0])
    item13.add(sort_items[2])
    item13 = frozenset(item13)

    item23 = set()
    item23.add(sort_items[1])
    item23.add(sort_items[2])
    item23 = frozenset(item23)

    conf12_3 = count / frequent_pairs[item12]
    conf13_2 = count / frequent_pairs[item13]
    conf23_1 = count / frequent_pairs[item23]
    print(f'{item1}\t{item2}\t{item3}\t{count}\t{conf12_3:.4f}\t{conf13_2:.4f}\t{conf23_1:.4f}')
