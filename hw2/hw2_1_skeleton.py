import sys
import math

from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

def dist(x, y):
    """
    INPUT: two points x and y
    OUTPUT: the Euclidean distance between two points x and y

    DESCRIPTION: Returns the Euclidean distance between two points.
    """
    return 


def parse_line(line):
    """
    INPUT: one line from input file
    OUTPUT: parsed line with numerical values
    
    DESCRIPTION: Parses a line to coordinates.
    """
    return 


def pick_points(k):
    """
    INPUT: value of k for k-means algorithm
    OUTPUT: the list of initial k centroids.

    DESCRIPTION: Picks the initial cluster centroids for running k-means.
    """
    return 


def assign_cluster(centroids, point):
    """
    INPUT: list of centorids and a point
    OUTPUT: a pair of (closest centroid, given point)

    DESCRIPTION: Assigns a point to the closest centroid.
    """
    return


def compute_diameter(cluster):
    """
    INPUT: cluster
    OUTPUT: diameter of the given cluster

    DESCRIPTION: Computes the diameter of a cluster.
    """
    return 


def kmeans(centroids):
    """
    INPUT: list of centroids
    OUTPUT: average diameter of the clusters

    DESCRIPTION: 
    Runs the k-means algorithm and computes the cluster diameters.
    Returns the average diameter of the clusters.

    You may use PySpark things at this function.
    """
    return


if __name__ == "__main__":
    """
    This is just an example of the main function.
    """
    k = int(sys.argv[2])
    centroids = pick_points(k)
    average_diameter = kmeans(centroids)
    print(average_diameter)