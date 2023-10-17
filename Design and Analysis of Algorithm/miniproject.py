# Mini Project - Different exact and approximation algorithms for Travelling-Sales-Person Problem

def tsp_dp(graph):
    n = len(graph)
    #creating a set of all nodes
    all_nodes = set(range(n))
    # A memoization dictionary to store computed subproblem results
    memo = {}

    def tsp(mask, current):
        if len(mask) == n :
            return graph[current][0]  #Return to starting node if all nodes are visited
        
        if(tuple(mask) , current) in memo:
            return memo[(mask, current)]    #Return memoised return if available
        
        min_cost = float('inf')             # initialising minimum cost to infinity
        for next_node in all_nodes - set(mask):
            #updated the set of visited node
            new_mask = mask + [next_node]
            cost = graph[current][next_node] + tsp(new_mask, next_node)
            # Finding the minimum cost among choices
            min_cost = min(min_cost, cost)

        memo[(tuple(mask), current)] = min_cost          #memoise the result
        return min_cost
    
    # Start with the first node and an empty set of visited nodes
    return tsp([0],0)

graph = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]

min_cost = tsp_dp(graph)
print("Minimum cost of grpah using Dynamic programming is: ",min_cost)

---------------------------------------------------------------------------------------------------
'''
#OUTPUT:
DAA> python mini.py
Minimum cost of grpah using Dynamic programming is:  73
'''
---------------------------------------------------------------------------------------------------
'''
Memoization is a method of storing the results of function calls, so that if the same function is called again with 
the same inputs, the previously computed result can be returned instead of recalculating it.

In the code, memo is a dictionary that serves as a memoization table. It stores the computed results for specific subproblems. The keys in 
this dictionary consist of a tuple (mask, current), where:

mask : is a list or tuple representing the set of visited cities (nodes).
current: is the index of the current city being considered.
'''


````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# I am using Nearesr Neighbor Heuritsic approximation algorithm to calculate minimum cost of tsp program

def tsp_nearest(graph):
    n = len(graph)
    unvisited = set(range(n))
    # Start from node 0 as the initial city
    path = [0]
    total_cost = 0

    while unvisited:
        #Get the last vivited node
        current_node = path[-1]

        #Finding nearest unvisited node using almbda function
        nearest = min(unvisited, key=lambda node: graph[current_node][node])
        # Adding the nearest neighbor to the tour
        path.append(nearest)
        #update the total cost
        total_cost += graph[current_node][nearest]
        #Mark nearest neighbor as visited 
        unvisited.remove(nearest)

    # Return to the starting node to complete the tour
    total_cost += graph[path[-1]][0]

    # Check if a valid tour was found
    if len(path) != n + 1 or path[-1] != 0:
        return None, None  # Invalid tour, return None

    return path, total_cost


graph = [
    [0, 29, 20, 21],
    [29, 0, 15, 17],
    [20, 15, 0, 28],
    [21, 17, 28, 0]
]

path, min_cost = tsp_nearest(graph)

if path is None: 
    print("No valid TSP path is found")
else:
    print("Approximate Tsp Path", path)
    print("Approximate min_cost", min_cost)

---------------------------------------------------------------------------------------------------
#OUTPUT:
DAA> python mini.py
No valid TSP path is found
