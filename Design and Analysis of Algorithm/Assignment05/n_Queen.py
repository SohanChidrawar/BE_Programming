def safe(arr,x,y,n):
    #checking if queen in same column
    for row in range(x):
        if arr[row][y] == 1:  
            return False

    #checking queen in left diagonal 
    row = x
    col = y
    while row>=0 and col>=0:
        if arr[row][col] == 1:
            # If a queen is already present in the diagonal, return False
            return False
        row = row-1
        col = col-1
        
    #checking queen in right diagonal
    row = x
    col = y
    while row>=0 and col<n:
        if arr[row][col] == 1:
            # If a queen is already present in the diagonal, return False
            return False
        row = row-1
        col = col+1

    # if there is no conflict, it is safe to place the queen
    return True

def Nqueen(arr,x,n,count):
    #count = 0
    # If all queens are placed, increment the count of solutions and print the board
    if x>=n:
        count[0] +=1
        print(count[0], ":")
        for i in range(n):
            for j in range(n):
                print(arr[i][j], end=" ")
            print()
        print()
        return
    
    # placing queen in each column of the current row
    for col in range(n):
        if safe(arr,x,col,n):                # Check if it is safe to place queen at current position
            arr[x][col] = 1                  # Place queen at current position
            Nqueen(arr,x+1,n,count)          # recursive function
            arr[x][col] = 0                  # If no solution is found in this path, backtrack by removing the queen

    return

def main():
    n = int(input("Enter number of queens:  "))
    # Create an empty board of size n x n
    arr = [[0]* n for i in range(n)]
    count = [0]

    # Placing the queens on the board recursively
    Nqueen(arr,0,n,count)
    print("Number of possible solutions",count[0])

if __name__ == '__main__':
    main()


'''
Output:

Enter number of queens:  4
1:
0 1 0 0
0 0 0 1
1 0 0 0
0 0 1 0

2 :
0 0 1 0
1 0 0 0
0 0 0 1
0 1 0 0

Number of possible solutions 2

Enter number of queens:  3
Number of possible solutions 0

Enter number of queens:  5
1 :
1 0 0 0 0
0 0 1 0 0
0 0 0 0 1
0 1 0 0 0
0 0 0 1 0

2 :
1 0 0 0 0
0 0 0 1 0
0 1 0 0 0
0 0 0 0 1
0 0 1 0 0

3 :
0 1 0 0 0
0 0 0 1 0
1 0 0 0 0
0 0 1 0 0
0 0 0 0 1

4 :
0 1 0 0 0
0 0 0 0 1
0 0 1 0 0
1 0 0 0 0
0 0 0 1 0

5 :
0 0 1 0 0
1 0 0 0 0
0 0 0 1 0
0 1 0 0 0
0 0 0 0 1

6 :
0 0 1 0 0
0 0 0 0 1
0 1 0 0 0
0 0 0 1 0
1 0 0 0 0

7 :
0 0 0 1 0
1 0 0 0 0
0 0 1 0 0
0 0 0 0 1
0 1 0 0 0

8 :
0 0 0 1 0
0 1 0 0 0
0 0 0 0 1
0 0 1 0 0
1 0 0 0 0

9 :
0 0 0 0 1
0 1 0 0 0
0 0 0 1 0 
1 0 0 0 0
0 0 1 0 0

10 :
0 0 0 0 1
0 0 1 0 0
1 0 0 0 0
0 0 0 1 0
0 1 0 0 0

Number of possible solutions 10

'''
