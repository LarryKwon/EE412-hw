import sys
import math
import re


from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
all_points = []
clusters = {}

def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y
    DESCRIPTION: Returns the Euclidean distance between two points.
    """
    square_sum = 0
    # print('x: ', x)
    # print('y: ',y)
    for i in range(len(x)):
        square_sum += (x[i] - y[i])**2
    return math.sqrt(square_sum)


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values
    
    DESCRIPTION: Parses a line to coordinates.
    """
    # print(line)
    coordinates = [float(coor) for coor in re.split('[^\d+.\d+]+',line) if coor];
    # print(len(coordinates))
    return coordinates


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.
    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """
    k_initial_point = []
    k_initial_point.append(all_points[0])
    for i in range(k-1):
        maximum_minimum_point_index = -1;
        maximum_minimum_distance = 0;
        for j in range(len(all_points)):
            minimum = math.inf
            for initial_point_index in k_initial_point:
                distance = dist(initial_point_index, all_points[j]);
                if(minimum > distance):
                    minimum = distance
            if minimum > maximum_minimum_distance:
                maximum_minimum_distance = minimum;
                maximum_minimum_point_index = j
        k_initial_point.append(all_points[maximum_minimum_point_index])
    
    return k_initial_point


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """
    closest_distance = math.inf
    closest_distance_centroid_index = None
    for centroid_index in range(len(centroids)):
            distance = dist(centroids[centroid_index], point)
            if distance < closest_distance:
                closest_distance = distance
                closest_distance_centroid_index = centroid_index
    return (closest_distance_centroid_index, point)


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    # print(cluster);
    cluster_points = list(cluster);
    max_distance = 0
    farthest_points = None
    for i in range(len(cluster_points)):
        for j in range(i + 1, len(cluster_points)):
            d = dist(cluster_points[i], cluster_points[j])
            if d > max_distance:
                max_distance = d
                farthest_points = (cluster_points[i], cluster_points[j])
    return max_distance


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters

    DESCRIPTION: 
    Runs the k-means algorithm and computes the cluster diameters.
    Returns the average diameter of the clusters.

    You may use PySpark things at this function.
    """
    lines = sc.textFile(sys.argv[1])
    points = lines.map(parse_line)
    points = points.map(lambda point: assign_cluster(centroids, point))
    # pints_list = points.collect()
    # pionts.saveAsTextFile(sys.argv[3])
    # print(pints_list)
    diameters = points.groupByKey().mapValues(compute_diameter)
    # diameters.saveAsTextFile(sys.argv[3])

    sums = diameters.map(lambda x: x[1]).reduce(lambda x, y: x + y)
    count = diameters.count()
    average = sums / count

    return average


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    file_name = sys.argv[1]
    
    with open(file_name,'r') as file:
        for line in file:
            all_points.append(parse_line(line))
    # print(all_points)
    k = int(sys.argv[2])
    centroids = pick_points(k)
    # print(centroids)
    average_diameter = kmeans(centroids)
    print(average_diameter)