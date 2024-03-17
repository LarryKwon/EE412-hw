import sys
import math
import re
import numpy as np
from pyspark import SparkConf, SparkContext

def power_iteration(beta, matrix,random_walk,num_iterations):
    n = matrix.shape[0]
    v = np.ones(n).reshape(n,-1)
    v = v/n
    eigenvalue = 0

    for i in range(num_iterations):
        v = beta * np.dot(matrix, v) + (1-beta) * random_walk
    return v

def initialize_matrix(elem):
    from_ = elem[0]
    to_ = elem[1]
    count = len(to_)
    return [(from_, elem_to_, count) for elem_to_ in to_]

conf = SparkConf()
sc = SparkContext(conf=conf)

file_name = sys.argv[1]

lines = sc.textFile(sys.argv[1])
nodes = lines.map(lambda line: [int(value)
                                  for value in re.split(r'[^\d]+', line) if value])
nodes = nodes.map(lambda list_: (list_[0],list_[1])).distinct().groupByKey()
nodes = nodes.flatMap(initialize_matrix)

matrix = nodes.map(lambda elem: (elem[1]-1,elem[0]-1,1/elem[2]))
edges = nodes.map(lambda elem: (elem[0],elem[1])).sortByKey()
outdegress = nodes.map(lambda elem: (elem[0], elem[2])).distinct().sortByKey()
# print('edges', edges.collect().sort(key=lambda x: (x[0], x[1])))
# print('----------------------------')
# print('outdegree',outdegress.collect().sort(key=lambda x: (x[0], x[1])))
pageRank = np.ones(1000)/1000
initial_vec = []

for i in range(0, len(pageRank)):
    initial_vec.append((i,1/1000))

random_walk = sc.parallelize(initial_vec)

beta = 0.9
for i in range(50):
    computed_matrix = matrix.map(lambda elem: (elem[0], pageRank[elem[1]] * elem[2] * beta)).reduceByKey(lambda n1, n2: n1 + n2)
    computed_matrix_list = computed_matrix.collect()
    # if(i==0):
        # print(len(computed_matrix_list))
    for row in computed_matrix_list:
        pageRank[row[0]] = row[1] + (1-beta) * 1/1000

# print(len(pageRank))
# pageRank = pageRank[1:]
# print(pageRank)
sorted_indices = np.argsort(pageRank)[::-1]  # 내림차순으로 정렬된 인덱스 배열

# 상위 10개의 값과 인덱스 출력
top_10_indices = sorted_indices[:10]

for index in top_10_indices:
    print(f'{index+1}\t{pageRank[index]}')

