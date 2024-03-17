import re
import sys
from pyspark import SparkConf, SparkContext

p = re.compile('^[a-zA-Z]')
small_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']


conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile(sys.argv[1])

words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))
lower_words = words.map(lambda w: w.lower())
alphabet_words = lower_words.filter(
    lambda w: p.match(w))
pairs = alphabet_words.map(lambda w: (w, 1))
unique_word = pairs.groupByKey().map(lambda k: k[0])
letter_pairs = unique_word.map(lambda k: (k[0], 1))
counts = letter_pairs.reduceByKey(lambda n1, n2: n1 + n2)

zero_count_letter_pairs = sc.parallelize(
    map(lambda x: (x, 0), small_letters))
result = counts.union(zero_count_letter_pairs)

result = result.reduceByKey(lambda n1, n2: n1 + n2).sortByKey().map(
    lambda x: f'{x[0]}\t{x[1]}')

result.saveAsTextFile(sys.argv[2])

result_list = result.collect()
for e in result_list:
    print(e)

# counts = result.reduceByKey(lambda n1, n2: n1 + n2)
# counts.saveAsTextFile(sys.argv[2])
# sc.stop()
