import re
import sys
from pyspark import SparkConf, SparkContext


def mapping_friends(lines: str):
    friends = [int(value) for value in re.split(r'[^\d]+', lines) if value]
    me = friends.pop()
    return (me, friends)


def combinations(arr):
    all_pairs = [(arr[i], arr[j]) for i in range(len(arr))
                 for j in range(i + 1, len(arr))]

    return all_pairs


def sort_friends(elem):
    friends = elem[1]
    friends.sort()
    return (elem[0], friends)


def custom_sort(item):
    ((user1, user2), count) = item
    return (-count, user1, user2)


conf = SparkConf()
sc = SparkContext(conf=conf)

### make friends rdd (me, [friends])
lines = sc.textFile(sys.argv[1])
friends = lines.map(lambda line: [int(value)
                                  for value in re.split(r'[^\d]+', line) if value])
pairs = friends.map(
    lambda list: (list[0], list[1:])).map(sort_friends)

### make each friend pairs from friends rdd (me, friend)
originals = pairs.flatMap(
    lambda elems: [(elems[0], elem) for elem in elems[1]])

### make candidates who may be potential friend
candidates = pairs.flatMap(lambda elems: [(elems[1][i], elems[1][j]) for i in range(len(elems[1]))
                                          for j in range(i + 1, len(elems[1]))])

### remove the current friend pairs from candidates
results = candidates.subtract(originals)

### reducing by key leads the total sum of potential friends
result_list = results.map(lambda elem: ((elem[0], elem[1]), 1)).reduceByKey(
    lambda n1, n2: n1+n2).sortBy(custom_sort).take(10)

for result in result_list:
    print(f'{result[0][0]}\t{result[0][1]}\t{result[1]}')
