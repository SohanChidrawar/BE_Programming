# RECURSIVE MANNER

def fibonacci(n):
    if n<= 0 :                                  # Base case: F(0) = 0
        return 0
    if n == 1:                                  # Base case: F(1) = 1
        return 1
    else:
        # Recursive case: F(n) = F(n-1) + F(n-2)
        return fibonacci(n-1) + fibonacci(n-2)
    
# Input the value of n from the user
n = int(input("Enter value of n: "))

if  n < 0:
    print("Enter a non-negative integer")
else:
    result = fibonacci(n)                        # Call the fibonacci function to calculate the Fibonacci number
    print("The fibonacci number is",result)


# ----------------------------------------------------------------------------------------- #

# OUTPUT
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 3
The fibonacci number is 2
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 10
The fibonacci number is 55
