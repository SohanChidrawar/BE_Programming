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

#-----------------------------------------------------------------------------------------------------------------
'''
EXPLAINATION

The provided Python code is an implementation of the Quick Sort algorithm with two variants: deterministic Quick Sort and randomized Quick Sort. 
The code generates a random list of numbers and analyzes the execution time for each variant. Here's a detailed explanation of the code:

Import Libraries:
random: This library is used for generating random numbers.
time: This library is used to measure the execution time of sorting algorithms.
Define Deterministic Quick Sort:

deterministic_quick_sort(arr): This function implements the deterministic Quick Sort algorithm.
It takes an input list arr and sorts it in ascending order using the Quick Sort algorithm.
The pivot is always chosen as the first element of the input list.
The function recursively divides the list into two sublists: elements less than or equal to the pivot (left) and elements greater than the pivot (right).
Define Randomized Quick Sort:

randomized_quick_sort(arr): This function implements the randomized Quick Sort algorithm.
It takes an input list arr and sorts it using the Quick Sort algorithm with a random pivot selection.
The pivot is selected randomly from the input list to avoid worst-case scenarios.
The function recursively divides the list into two sublists as in the deterministic version.

Generate a Random List:
size: This variable specifies the size of the random list (the number of elements).
random_list: A random list of size elements is generated with values between 1 and 10,000 using the random.randint function.

Analyze Deterministic Quick Sort:
The code measures the execution time of the deterministic Quick Sort by making a copy of the random list (random_list.copy()), sorting it, and recording 
the time taken. This time measurement helps evaluate the efficiency of the algorithm.

Analyze Randomized Quick Sort:
Similarly, the code measures the execution time of the randomized Quick Sort. It copies the random list, sorts it using the randomized variant, and records
the time taken.

Print Results:
The code prints the size of the list, the execution time for the deterministic Quick Sort, and the execution time for the randomized Quick Sort.
In summary, this code provides a comparison of two Quick Sort variants, one with a deterministic pivot selection strategy and the other with a randomized 
pivot selection strategy. It measures the time taken for each variant to sort a random list of numbers and prints the results, allowing you to compare their
performance. The randomized variant is expected to perform more consistently and efficiently, especially in cases with structured or partially sorted data.


'''
