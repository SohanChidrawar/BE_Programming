# RECURSIVE MANNER

def fibonacci(n):
    if n<= 0 :
        return 0
    if n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)
    
n = int(input("Enter value of n: "))

if  n < 0:
    print("Enter a non-negative integer")
else:
    result = fibonacci(n)
    print("The fibonacci number is",result)


# ----------------------------------------------------------------------------------------- #

# OUTPUT
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 3
The fibonacci number is 2
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 10
The fibonacci number is 55
