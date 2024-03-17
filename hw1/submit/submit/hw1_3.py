import re
import sys
import time
import math
import numpy as np

file_name = sys.argv[1]
band = 6
row = 20


def characteristic_matrix(characteristic_matrix: dict):
    for shingle in characteristic_matrix:
        one_count = 0
        for docu in characteristic_matrix[shingle]:
            if (characteristic_matrix[shingle][docu] == 1):
                one_count += 1
        if (one_count > 100):
            print(f'{shingle} : {one_count}')


def is_prime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def next_prime(num):
    while True:
        if is_prime(num):
            return num
        num += 1
    return num


def init():
    all_docs = {}
    with open(file_name, 'r') as file:
        for line in file:
            parts = line.split(' ', 1)
            content = parts[1]
            content = re.sub(r'[^a-zA-Z\s]', '', content).lower()
            content = re.sub(r'\s+', ' ', content)
            all_docs[parts[0]] = content
    return all_docs


def k_shingling(docs: dict, shingle_length=3):
    shingles = set()
    for docu in docs:
        content = docs[docu]
        for i in range(len(content) - shingle_length + 1):
            shingles.add(content[i:i+shingle_length])
    return list(shingles)


def make_characteristic_matrix(docs: dict, shingles):
    characteristic_matrix = {}

    column = [0] * len(shingles)
    for docu in docs:
        column_copy = column.copy()
        characteristic_matrix[docu] = column_copy
        for i in range(len(shingles)):
            if shingles[i] in all_docs[docu]:
                characteristic_matrix[docu][i] = 1
        # print(f'{docu} finish')
    return characteristic_matrix


def make_hash_functions(shingles):
    hash_functions = []
    n = len(shingles)
    c = next_prime(n)

    random_coef = np.random.randint(1, c, size=(2, band * row))
    for i in range(band*row):
        def funcC(j):
            def func(x): return (
                random_coef[0][j]*x + random_coef[1][j]) % c
            return func
        # print((random_coef[0][i]*1 + random_coef[1][i]) % c)
        hash_functions.append(funcC(i))
    return hash_functions


def print_signature_matrix(signature_matrix: dict):
    for docu in signature_matrix:
        print(docu, end=": ")
        for value in signature_matrix[docu]:
            print(value, end=" ")
        print('\n')


def make_signature_matrix(characteristic_matrix: dict, hash_functions, all_docs: dict, shingles: dict):

    signature_matrix = {}
    for docu in all_docs:
        signature_matrix[docu] = []
    index = 0
    hash_function_start_time = time.time()
    for hash_function in hash_functions:
        # print(hash_function(1))
        for docu in characteristic_matrix:
            min_value = None
            for i in range(len(characteristic_matrix[docu])):
                if (characteristic_matrix[docu][i] == 0):
                    min_value == None
                else:
                    if (min_value == None):
                        min_value = hash_function(i)
                    else:
                        if min_value > hash_function(i):
                            min_value = hash_function(i)
            signature_matrix[docu].append(min_value)
        hash_function_end_time = time.time()
        # print(f'{index}: {hash_function_end_time - hash_function_start_time}')
        index += 1
        # print_signature_matrix(signature_matrix)

    return signature_matrix


def make_band_hash_functions(shingles, band, row):
    hash_functions = []
    n = len(shingles)
    c = next_prime(n)

    random_coef = np.random.randint(1, c, size=(band, row))
    for i in range(band):
        def funcC(j):
            def func(x): return sum(
                [random_coef[j][k] ^ x[k] for k in range(row)]) % c
            return func
        # print((random_coef[0][i]*1 + random_coef[1][i]) % c)
        hash_functions.append(funcC(i))
    return hash_functions, c    
    

def make_band_hash_function(shingles, band, row):
    hash_functions = []
    n = len(shingles)
    c = next_prime(n)
    # hash_functions.append(hash)

    random_coef = np.random.randint(1, c, size=(2, band))
    for i in range(band*row):
        def funcC(j):
            def func(x): return (
                random_coef[0][j]*sum(x) + random_coef[1][j]) % c
            return func
        # print((random_coef[0][i]*1 + random_coef[1][i]) % c)
        hash_functions.append(funcC(i))
    return hash_functions


def make_similar_pair(signature_matrix: dict, hash_functions, c):
    similar_pair = set()
    for i in range(band):
        buckets = {}
        for docu in signature_matrix:
            signature = tuple(signature_matrix[docu][i*band:i*band+row])
            hash_values = tuple([hash_functions[i](value) for value in signature])
            if hash_values in buckets:
                buckets[hash_values].append(docu)
            else:
                buckets[hash_values] = []
                buckets[hash_values].append(docu)            
            # if signature in buckets:
            #     buckets[signature].append(docu)
            # else:
            #     buckets[signature] = []
            #     buckets[signature].append(docu)

        for bucket in buckets:
            if(len(buckets[bucket]) > 1):
                similars = [frozenset([buckets[bucket][i], buckets[bucket][j]]) for i in range(len(buckets[bucket])) for j in range(i + 1, len(buckets[bucket]))]
                for similar in similars:
                    similar_pair.add(similar)
    return similar_pair


start_time = time.time()
all_docs = init()
shingles = k_shingling(all_docs)
shingle_time = time.time()
# print(f'shingle elapsed time: {shingle_time-start_time}')

hash_functions = make_hash_functions(shingles)
hash_function_gen = time.time()
# print(f'hash function elapsed time: {hash_function_gen-start_time}')

band_hash_functions, c = make_band_hash_functions(shingles, band, row)
# for hash_function in band_hash_functions:
#     print(hash_function([1]*20))

characteristic_matrix = make_characteristic_matrix(all_docs, shingles)
characteristic_matrix_time = time.time()
# print(
#     f'characteristic matrix elapsed time: {characteristic_matrix_time-start_time}')

signature_matrix = make_signature_matrix(
    characteristic_matrix, hash_functions, all_docs, shingles)
# print_signature_matrix(signature_matrix)
signature_matrix_time = time.time()
# print(f'signature_matrix elapsed time: {signature_matrix_time-start_time}')


similar_pair = make_similar_pair(signature_matrix, hash_functions, c)
similar_pair_time = time.time()
# print(f'similar pair elapsed time: {similar_pair_time-start_time}')

similar_pair_list = list(similar_pair)
for pair in similar_pair_list:
    pair_list = list(set(pair))
    print(f'{pair_list[0]}\t{pair_list[1]}')