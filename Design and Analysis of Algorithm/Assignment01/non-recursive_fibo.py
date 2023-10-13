# NON-RECURSIVE MANNER

def fibonacci(n):
    if n<= 0:
        return 0
    if n == 1:
        return 1
    else:
        a,b = 0,1               # Initialize the first two Fibonacci numbers
        for i in range(2,n+1):  # Start the loop from the third Fibonacci number
            a,b = b , a + b     # Update a and b to calculate the next Fibonacci number
        return b                # Return nth fibonacci number

# Input the value of n from the user   
n = int(input("Enter value of n: "))
if n<0:
    print("Enter non- negative number")
else:
    result = fibonacci(n)       # Call the fibonacci function to calculate the Fibonacci number
    print("The Fibonacci number is ",result)


# ---------------------------------------------------------------------------------------------- #

# OUTPUT
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 3
The Fibonacci number is  2
 MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 10
The Fibonacci number is  55
