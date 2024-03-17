import numpy as np
import sys
import re


# Graph class
class Graph:

    # Store the graph as set of adjacency lists of each nodes
    # Assume undirected graph
    # For example, if we have node 1, node 2 and edge (1, 2), (1, 3),
    # adj_list = {1: [2, 3], 2: [1], 3: [1]} 
    def __init__(self):
        self.adj_list = {}

    # Add an edge to the graph (i.e., update adj_list)
    def add_edge(self, u, v):
        # print(u,v)
        if  u not in self.adj_list:
            self.adj_list[u] = []
        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    # Return the neighbors of a node
    def neighbors(self, node):
        return self.adj_list[node]

    def dfs(self, start, visited=None, walk=None,stack=None, max_length=5):
        if visited is None:
            visited = set()
        if walk is None:
            walk = []
        if stack is None:
            stack = []
            
            
        visited.add(start)
        walk.append(start)
        stack.append(start)
        
        # print(walk,visited, stack)
        if len(walk) >= max_length:
            return walk[:max_length]
        unvisited_neighbors = [neighbor for neighbor in self.neighbors(start) if neighbor not in visited]
        
        if(unvisited_neighbors):
            for neighbor in unvisited_neighbors:
                if neighbor not in visited:
                    self.dfs(neighbor, visited, walk,stack)
                    if len(walk) >= max_length:
                        return walk
        else:
            stack.pop()
            previous = stack.pop()
            self.dfs(previous,visited,walk,stack)
            
            
        return walk


# This random_walk funtion is a hint 
# Refer to this function to implement the node2vec_walk function below
def random_walk(graph, start, length=5):
    # Random Walk
    walk = [start]
    for _ in range(length - 1):
        neighbors = graph.neighbors(walk[-1])
        if not neighbors:
            break
        walk.append(np.random.choice(neighbors))
    return walk

# Implement node2vec algorithm in DFS
# The length of the walk is fixed to 5
# When sampling next node, visit the node with the smallest index
# Note that it returns the trajectory of walker
# so that same node can be visited multiple times
def node2vec_walk_dfs(graph: Graph, start, length=5):
    # print(start)
    return graph.dfs(start)



# Train W1, W2 matrices using Skip-Gram
# The window size of fixed to 2, which means you should check each 2 nodes before and after the center node.  
# - Ex. Assume we have walk sequence [1, 2, 3, 4, 5].
# - If center node is 3, we should consider [1, 2, 3, 4, 5] as the context nodes.
# - If center node is 2, we should consider [1, 2, 3, 4] as the context nodes.
# Repeat the training process for 3 epochs, with learning rate 0.01
# Use softmax function when computing the loss
def train_skipgram(walks, n_nodes, dim=128, lr=0.01, window=2, epochs=3):
    W1 = np.random.randn(n_nodes, dim)
    W2 = np.random.randn(dim, n_nodes)

    for _ in range(epochs):
        for walk in walks:
            for i, target in enumerate(walk):
                context = walk[max(0, i - window):i] + walk[i + 1:i+window+1]
                for c in context:                    
                    z = W1.T[:,target-1]
                    s = W2.T @ z
                    y = np.exp(s- np.max(s)) / np.sum(np.exp(s - np.max(s)))  # softmax
                    ans = np.zeros(n_nodes)
                    ans[c-1] = 1
                    error = y - ans
                    # W2 -= lr * np.outer(z, error)
                    W2_grad = z.reshape(-1,1) * error.reshape(1,-1)
                    W1_grad = np.matmul(error, W2.T)
                    W1[target-1,:] -= lr * W1_grad
                    W2 -= lr * W2_grad
    return W1

# You can freely define your functions/classes if you want
def your_function():
    pass


# Main function
def main():

    # Don't change this code
    # This will guarantee the same output when we test your code
    np.random.seed(1116)


    # Create graph
    graph = Graph()
    file_path = sys.argv[1]
    
    directed_edges = []
    
    with open(file_path, 'r') as file:
        for line in file:
            items = [int(item) for item in re.split(r'[^\w]+', line) if item]
            directed_edges.append((items[0],items[1]))
    undirected_edges = list({tuple(sorted(t)) for t in directed_edges})
    # Edges list
    # Note that the edges are undirected, and node idx starts with 1
    # ex. edges = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]  
    edges = sorted(undirected_edges, key = lambda x: (x[0],x[1]))
    # print(edges)


    # Parse edges from the command line file path ======================
    # Implement your code here
    
    # ====================================================================
    

    # Update graph
    for edge in edges:
        # print(edge)
        graph.add_edge(*edge)
    
    # Generate random walks on DFS/BFS
    walks_dfs = [node2vec_walk_dfs(graph, node) for node in sorted(graph.adj_list)]

    # print("=====================================")
    # print("Test of node2vec_walk_dfs()")
    # for i, item in enumerate(walks_dfs):
    #     print("node : ", i+1)
    #     print("walk : ", item)
    # print("=====================================\n")


    # Train Skip-Gram on DFS/BFS
    embeddings_dfs = train_skipgram(walks_dfs, len(graph.adj_list))


    # Print the embeddings ===========================================
    # Implement your code here
    
    # print("=====================================")
    # print("Test of train_skipgram()")
    # print("First element of node 5's embedding: ", )
    print(f"{embeddings_dfs[4][0]:.5f}")
    print(f"{embeddings_dfs[9][0]:.5f}")
    # print("=====================================\n")


    # ===============================================================


if __name__ == "__main__":
    main()
