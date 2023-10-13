# NON-RECURSIVE MANNER

def fibonacci(n):
    if n<= 0:
        return 0
    if n == 1:
        return 1
    else:
        a,b = 0,1
        for i in range(2,n+1):
            a,b = b , a + b
        return b
    
n = int(input("Enter value of n: "))
if n<0:
    print("Enter non- negative number")
else:
    result = fibonacci(n)
    print("The Fibonacci number is ",result)


# ---------------------------------------------------------------------------------------------- #

# OUTPUT
MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 3
The Fibonacci number is  2
 MESCOE\Desktop\BE\7sem\DAA> python fibo.py
Enter value of n: 10
The Fibonacci number is  55
