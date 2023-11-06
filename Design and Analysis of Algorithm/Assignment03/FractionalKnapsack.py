def fractionalknapsack(items,capacity):
    #sort item by their value to weight ratio in descending order
    items.sort(key = lambda x: x[1]/x[2], reverse = True)

    total_value = 0        # Total value in knapsack
    knapsack = []          #List to store selected value

    for item in items:
        item_name, item_value, item_weight = item

        if capacity >= item_weight:
            knapsack.append((item_name,item_weight))
            total_value += item_value
            capacity -= item_weight
        
        else:
             # If only a fraction of the item can be added, add that fraction
             fraction = capacity / item_weight
             knapsack.append((item_name, capacity))
             total_value += item_value * fraction
             break
        
    return total_value, knapsack

if __name__ == "__main__":
    # items = [("Item1", 60, 10), ("Item2", 100, 20), ("Item3", 120, 30)]
    
    # capacity = 50
    items = [("Item1",10,2), ("Item2",5,3), ("Item3",15,5), ("Item4",7, 7), ("Item5", 6,1)]
    capacity = 10
    
    total_value, selected_items = fractionalknapsack(items,capacity)

    print("Maximum value can be obtained: ",total_value)
    print("Selected item in the knapsack: ")
    for item in selected_items:
        item_name, item_weight = item
        print("",item_name,"- Weight:",item_weight)

     


# ----------------------------------------------------------------------------------------------------------- #
'''
 OUTPUT:

MESCOE\Desktop\BE\7sem\DAA> python 3.py
Maximum value can be obtained:  34.333333333333336
Selected item in the knapsack: 
 Item5 - Weight: 1
 Item1 - Weight: 2
 Item3 - Weight: 5
 Item2 - Weight: 2
'''

# The code items.sort(key=lambda x: x[1] / x[2], reverse=True) is sorting the items list based on a custom sorting key using a lambda function.
# Here's the breakdown of what it means:

# items.sort: This is a method used to sort the items list in-place. It will reorder the items within the list according to the sorting criteria
# specified.

# key=lambda x: x[1] / x[2]: This is a lambda function that calculates the value-to-weight ratio for each item x. In the lambda function:

# x[1] refers to the second element of each item, which is typically the item's value.
# x[2] refers to the third element of each item, which is typically the item's weight.
# x[1] / x[2] calculates the value-to-weight ratio for the item x. It's used as the key for sorting.
# reverse=True: This parameter specifies that the sorting should be in descending order. In other words, it will sort the items with the highest
# value-to-weight ratio first, effectively sorting items from the most valuable to the least valuable per unit weight.
