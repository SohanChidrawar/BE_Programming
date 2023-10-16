def knapsack_dp(values, weights, capacity):
    n = len(values)

    #Initialise a 2D table to store the maximum values for different subproblem
    dp = [[0 for _ in range(capacity+1)] for _ in range(n+1)]

    #Building dp table in bottom up approach

    for i in range(n+1):
        for w in range(capacity+ 1):
            #base case
            if i == 0 or w == 0:
                dp[i][w]=0
            elif weights[i - 1]<=w:
                #include the current item or exclude it, take maximum value
                dp[i][w] = max(values[i - 1]+ dp[i-1][w - weights[i-1]], dp[i-1][w])
            else:
                # if the current weight is more than current capacity, skip it
                dp[i][w] = dp[i - 1][w]

    #Backtrack to find the selected item
    selected_items = []
    i, w = n, capacity
    while i>0 and w>0 :
        if dp[i][w] != dp[i - 1][w]:
            # If the value at (i, w) differs from the value above (i-1, w),
            # it means the current item (i-1) was included in the knapsack.
            selected_items.append(i-1)
            w -= weights[i-1]           # Reduce the remaining capacity by the item's weight
        i -=1

    return dp[n][capacity], selected_items

if __name__ == "__main__":
    values = [60, 100, 120]
    weights = [10, 20, 30]
    capacity = 50

    max_value, selected_items = knapsack_dp(values, weights, capacity)

    print("Maximum value that can be obtained :", max_value)
    print("Selected items:",selected_items)




# ------------------------------------------------------------------------------------------------------------------- #
'''
OUTPUT:

MESCOE\Desktop\BE\7sem\DAA> python 4.py   
Maximum value that can be obtained : 220
Selected items: [2, 1]

'''
