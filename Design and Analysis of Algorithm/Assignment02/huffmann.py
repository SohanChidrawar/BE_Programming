# A Huffman Tree Node 
import heapq 

class node: 
	def __init__(self, freq, symbol, left=None, right=None): 
		# frequency of symbol 
		self.freq = freq 

		# symbol name (character) 
		self.symbol = symbol 

		# node left of current node 
		self.left = left 

		# node right of current node 
		self.right = right 

		# tree direction (0/1) 
		self.huff = '' 

	def __lt__(self, nxt): 
		return self.freq < nxt.freq 


# utility function to print huffman codes for all symbols in the newly 
# created Huffman tree
 
def printNodes(node, val=''): 

	# huffman code for current node 
	newVal = val + str(node.huff) 

	# if node is not an edge node 
	# then traverse inside it 
	if(node.left): 
		printNodes(node.left, newVal) 
	if(node.right): 
		printNodes(node.right, newVal) 

		# if node is edge node then 
		# display its huffman code 
	if(not node.left and not node.right): 
		print(f"{node.symbol} -> {newVal}") 


# Dynamic input for characters and frequencies
num_chars = int(input("Enter the number of characters: "))
chars = []
freq = []
for i in range(num_chars):
    char = input(f"Enter character {i + 1}: ")
    frequency = int(input(f"Enter frequency for {char}: "))
    chars.append(char)
    freq.append(frequency)
	
# list containing unused nodes 
nodes = [] 

# converting characters and frequencies 
# into huffman tree nodes 
for x in range(len(chars)): 
	heapq.heappush(nodes, node(freq[x], chars[x])) 

while len(nodes) > 1: 

	# sort all the nodes in ascending order 
	# based on their frequency 
	left = heapq.heappop(nodes) 
	right = heapq.heappop(nodes) 

	# assign directional value to these nodes 
	left.huff = 0
	right.huff = 1

	# combine the 2 smallest nodes to create 
	# new node as their parent 
	newNode = node(left.freq+right.freq, left.symbol+right.symbol, left, right)

	heapq.heappush(nodes, newNode) 

# Huffman Tree is ready! 
printNodes(nodes[0]) 


# ----------------------------------------------------------------------------------------------------------------
'''
Output:

python huffmann.py
Enter the number of characters: 6
Enter character 1: a
Enter frequency for a: 5
Enter character 2: b
Enter frequency for b: 9
Enter character 3: c
Enter frequency for c: 12
Enter character 4: d
Enter frequency for d: 13
Enter character 5: e
Enter frequency for e: 16
Enter character 6: f
Enter frequency for f: 45
Huffman Codes:
f -> 0
c -> 100
d -> 101
a -> 1100
b -> 1101
e -> 111

'''
