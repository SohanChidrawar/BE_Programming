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
