import random
import time

# Deterministic Quick Sort
def deterministic_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        left = [x for x in arr[1:] if x <= pivot]
        right = [x for x in arr[1:] if x > pivot]
        return deterministic_quick_sort(left) + [pivot] + deterministic_quick_sort(right)

# Randomized Quick Sort
def randomized_quick_sort(arr):
    if len(arr) <= 1:
        return arr
    else:
        # Randomly select a pivot index
        pivot_index = random.randint(0, len(arr) - 1)
        pivot = arr[pivot_index]
        # Divide elements into two subarrays based on the pivot
        left = [x for i, x in enumerate(arr) if x <= pivot and i != pivot_index]
        right = [x for i, x in enumerate(arr) if x > pivot and i != pivot_index]
        # Recursively sort subarrays and concatenate them
        return randomized_quick_sort(left) + [pivot] + randomized_quick_sort(right)

# Generate a random list of numbers
size = 1000  # Change this to the desired size of the list
random_list = [random.randint(1, 10000) for _ in range(size)]

# Analyze deterministic Quick Sort
start_time = time.time()
sorted_list = deterministic_quick_sort(random_list.copy())
deterministic_time = time.time() - start_time

# Analyze randomized Quick Sort
start_time = time.time()
sorted_list = randomized_quick_sort(random_list.copy())
randomized_time = time.time() - start_time

# Print the results
print(f"Size of the list: {size}")
print(f"Deterministic Quick Sort Time: {deterministic_time:.6f} seconds")
print(f"Randomized Quick Sort Time: {randomized_time:.6f} seconds")



#--------------------------------------------------------------------------------------------------------------
'''
OUTPUT:

python 6.py
Size of the list: 1000
Deterministic Quick Sort Time: 0.015711 seconds
Randomized Quick Sort Time: 0.000000 seconds
'''
