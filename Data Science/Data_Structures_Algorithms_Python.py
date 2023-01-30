'''
Data Structures and Algorithms

1. Work with Linked Lists and Stacks and Understand Big O notation
2. Queues, Hash Tables, Trees, Graphs, and Recursion
3. Searching algorithms
4. Sorting algorithms

Algorithms: A set of instructions that solve a problem
1. Design
2. Code

Data Structures: Hold and manipulate data when we execute an algorithm
- Linked lists, stacks, queues, etc. 

Linked Lists: Sequence of data connected through links
- Composed of nodes. Each node has the data and a pointer to the next node.
- With the pointer the data doesn't need to be stored in contiguous blocks of memory; data can be located in any available memory address
- One link is a singly linked list. Two links in both directions are doubly linked lists

Can implement other data structures:
- Stacks
- Queues 
- Graphs
Access information by moving forward and backward

Methods:
- insert_at_beginning()
- remove_at_beginning()
- insert_at_end()
- remove_at_end()
- insert_at()
- remove_at()
- search()
'''

# In the video, you learned how to create singly linked lists by implementing the Node() class and the LinkedList() class.

# In this exercise, you will practice implementing both of these classes. In the first step, we will implement the Node() class, and in the second step the LinkedList() class. Are you ready?

class Node:
  def __init__(self, data):
    self.value = data
    # Leave the node initially without a next value
    self.next = None

class LinkedList:
  def __init__(self):
    # Set the head and the tail with null values
    self.head = None
    self.tail = None

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In the previous exercise, you learned how to implement a Node() and LinkedList() class.

# In this exercise, you will prepare the code for the insert_at_beginning() method to add a new node at the beginning of a linked list.

def insert_at_beginning(self, data):
    # Create the new node
    new_node = Node(data)
    # Check whether the linked list has a head node
    if self.head:
      # Point the next node of the new node to the head
      new_node.next = self.head
      self.head = new_node
    else:
      self.tail = new_node      
      self.head = new_node

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In the previous exercise, you learned how to insert a node at the beginning of a linked list.

# In this exercise, you will prepare the code for the remove_at_beginning() method. To do it, you will need to point the head of the linked list to the next node of the head.

class LinkedList:
  def __init__(self):
    self.head = None
    self.tail = None
    
  def remove_at_beginning(self):
    # The "next" node of the head becomes the new head node
    self.head = self.head.next

##################################################################################################################################################################
'''
Big O Notation

Measures the worst-case complexity of an algorithm
- Time complexity: Time taken to run completely
- Space complexity: Extra memory space

Uses mathematical expressions, not seconds/bytes: O(1), O(n), O(n^2), etc.

Input size vs. number of operations: n vs O(n...)
'''

# In this exercise, you will keep practicing your understanding of Big O notation.

# In the first step, you will create an algorithm that prints all the elements of the following list:

# colors = ['green', 'yellow', 'blue', 'pink']
# The algorithm will have an  complexity.

# In the second and third steps, you will calculate the complexity of two algorithms.

colors = ['green', 'yellow', 'blue', 'pink']

def linear(colors):
  # Iterate the elements of the list
  for color in colors:
    # Print the current element of the list
    print(color)	

linear(colors)

# The following algorithm shows you how to add a node at the beginning of a singly linked list using the Node() class and the insert_at_beginning() method. What is the complexity of this algorithm?

def insert_at_beginning(self,data):
    new_node = Node(data)
    if self.head:
        new_node.next = self.head
        self.head = new_node
    else:
        self.tail = new_node      
        self.head = new_node

# Complexity is O(1) because it is only checking a single element at the first position regardless of the size of the linked list

# The following algorithm searches for a given value within a linked list. What is its complexity?

# This method uses the Node() class and search() method.

def search(self, data):
    current_node = self.head
    while current_node:
        if current_node.data == data:
            return True
        else:
            current_node = current_node.next
    return False

# The algorithm contains one iterator that, in the worst-case visits all the nodes of the linked list. This amounts to a complexity of O(n) for the whole algorithm.

##################################################################################################################################################################
'''
Stacks

LIFO: Last in first out
- Think of a stack of books

Can only add to the top (pushing to the stack) or remove from the top (popping from the stack)
Can only read from the top of the stack (peeking)

Stacks- Real Uses
- Undo functionality (Undo removes the last action from the stack)
- Function calls (push block of memory, pop after execution ends)

Can be implemented using singly linked lists

Python has the LifoQueue object form the queue module to define stacks
'''
# In the last video, you learned how to implement stacks in Python. As you saw, stacks follow the LIFO principle; the last element inserted is the first element to come out.

# In this exercise, you will follow two steps to implement a stack with the push() operation using a singly linked list. You will also define a new attribute called size to track the number of items in the stack. You will start coding the class to build a Stack(), and after that, you will implement the push() operation.

# To program this, you will use the Node() class that has the following code:

class Node:
  def __init__(self, data):
    self.data = data
    self.next = None

class Stack:
  def __init__(self):
    # Initially there won't be any node at the top of the stack
    self.top = None
    # Initially there will be zero elements in the stack
    self.size = 0

    def push(self, data):
        # Create a node with the data
        new_node = Node(data)
        if self.top:
        new_node.next = self.top
        # Set the created node to the top node
        self.top = new_node
        # Increase the size of the stack by one
        self.size += 1

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will implement the pop() operation for a stack. pop() will be used to remove an element from the top of the stack. Again, we will consider the size attribute to know the number of elements in the stack.

class Stack:
  def __init__(self):
    self.top = None
    self.size = 0
    
  def pop(self):
    # Check if there is a top element
    if self.top is None:
      return None
    else:
      popped_node = self.top
      # Decrement the size of the stack
      self.size -= 1
      # Update the new value for the top node
      self.top = self.top.next
      popped_node.next = None
      return popped_node.data 

# The complexity of both pushing and popping is O(1) because they are only accessing a single element at the end, so the size of the stack does not matter.

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will work with Python's LifoQueue(). You will create a stack called my_book_stack to add books and remove them from it.

# Import the module to work with Python's LifoQueue
import queue

# Create an infinite LifoQueue
my_book_stack = queue.LifoQueue(maxsize=0)

# Add an element to the stack
my_book_stack.put("Don Quixote")

# Remove an element from the stack
my_book_stack.get()

##################################################################################################################################################################
##################################################################################################################################################################
'''
Queues, Hash Tables, Graphs and Trees, Recursion

Queues: FIFO (First in first out)
- Can only insert at the end (Enqueue)
- Can only remove at the head (Dequeue)

Printing tasks: Documents are printed in the order they are received

Queues can be implemented using linked lists and nodes

In Python, the queue module has the Queue and SimpleQueue classes
'''
# In the last video, you learned that queues can have multiple applications, such as managing the tasks for a printer.

# In this exercise, you will implement a class called PrinterTasks(), which will represent a simplified queue for a printer. To do this, you will be provided with the Queue() class that includes the following methods:

# enqueue(data): adds an element to the queue
# dequeue(): removes an element from the queue
# has_elements(): checks if the queue has elements. This is the code:
#     def has_elements(self):
#       return self.head != None
# You will start coding the PrinterTasks() class with its add_document() and print_documents() methods. After that, you will simulate the execution of a program that uses the PrinterTasks() class.

class PrinterTasks:
  def __init__(self):
    self.queue = Queue()
      
  def add_document(self, document):
    # Add the document to the queue
    self.queue.enqueue(document)
      
  def print_documents(self):
    # Iterate over the queue while it has elements
    while self.queue.has_elements():
      # Remove the document from the queue
      print("Printing", self.queue.dequeue())

printer_tasks = PrinterTasks()
# Add some documents to print
printer_tasks.add_document("Document 1")
printer_tasks.add_document("Document 2")
printer_tasks.add_document("Document 3")
# Print all the documents in the queue
printer_tasks.print_documents()

# The following code shows you how to enqueue() an element into a queue and dequeue() an element from a queue using singly linked lists. Can you calculate the complexity of both methods using Big O Notation?

def enqueue(self,data):
    new_node = Node(data)
    if self.head == None:
      self.head = new_node
      self.tail = new_node
    else:
      self.tail.next = new_node
      self.tail = new_node 

def dequeue(self):
    if self.head:
      current_node = self.head
      self.head = current_node.next
      current_node.next = None

      if self.head == None:
        self.tail = None

# Both of these are once again O(1) complexity because they only refer to the first or last element regardless of the queue size.

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will work with Python's SimpleQueue(). You will create a queue called my_orders_queue to add the orders of a restaurant and remove them from it when required.

import queue

# Create the queue
my_orders_queue = queue.SimpleQueue()

# Add an element to the queue
my_orders_queue.put("samosas")

# Remove an element from the queue
my_orders_queue.get()

##################################################################################################################################################################
# Hash Tables

# Stores a collection of items in key-value pairs
# In python we use dictionaries, same thing
# Can return the same output for different inputs

# You have been given a program that is supposed to iterate over the dishes of a menu, printing the name and its value.

# The dishes of the menu are stored in the following dictionary:

my_menu = {
  'lasagna': 14.75,
  'moussaka': 21.15,
  'sushi': 16.05,
  'paella': 21,
  'samosas': 14
}
# Testing the program, you realize that it is not correct.

# Correct the mistake
for key, value in my_menu.items():
  # Correct the mistake
  print(f"The price of the {key} is {value}.")

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# You are writing a program that iterates over the following nested dictionary to determine if the dishes need to be served cold or hot.

# Can you complete the program so that it outputs the following?

# Sushi is best served cold.
# Paella is best served hot.
# Samosa is best served hot.
# Gazpacho is best served cold.

# Iterate the elements of the menu
for dish, values in my_menu.items():
  # Print whether the dish must be served cold or hot
  print(f"{dish.title()} is best served {values['best_served']}.")

##################################################################################################################################################################
'''
Trees and Graphs

Trees are node-based data structures
Each node can have links to more than one node 
First node of the tree is the root
Nodes connected to the row above are the children of the above node
Trees can represent hierarchical relationships
- Structure of an HTML file, rival chess moves, etc.

Graphs are a set of modes/vertices and links/edges
Trees are a type of graph
Graphs can be directed (follow a specific direction) or undirected, weighted (numeric values associated with edges) or unweighted
User relationships in social networks, locations and distances, etc.
Searching and sorting algorithms

Trees must have connected nodes, graphs don't. Trees cannot have cycles, graphs can.
'''
# You have been given a program that is supposed to create the following binary tree:

# Testing it, you realize that the program is not correct. Could you correct it so that it works correctly?

class TreeNode:
  
  def __init__(self, data, left=None, right=None):
    # Correct the mistakes
    self.data = data
    self.left_child = left
    self.right_child = right

node1 = TreeNode("B")
node2 = TreeNode("C")
# Correct the mistake
root_node = TreeNode("A", node1, node2)

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# This exercise has two steps. In the first one, you will modify this code so that it can be used to create a weighted graph. To do this, you can use a hash table to represent the adjacent vertices with their weights. In the second step, you will build the following weighted graph:

class WeightedGraph:
  def __init__(self):
    self.vertices = {}
  
  def add_vertex(self, vertex):
    # Set the data for the vertex
    self.vertices[vertex] = []
    
  def add_edge(self, source, target, weight):
    # Set the weight
    self.vertices[source].append([target, weight])

my_graph = WeightedGraph()

# Create the vertices
my_graph.add_vertex('Paris')
my_graph.add_vertex('Toulouse')
my_graph.add_vertex('Biarritz')

# Create the edges
my_graph.add_edge('Paris', 'Toulouse', 678)
my_graph.add_edge('Toulouse', 'Biarritz', 312)
my_graph.add_edge('Biarritz', 'Paris', 783)

##################################################################################################################################################################
'''
Understanding Recursion

-A function calling itself
-Don't forget to define the base case or the code will run forever
-Computer uses a stack to keep track of the functions
    -Call stack

Dynamic programming:
-Optimization technique
-Mainly applied to recursion
-Can reduce the complexity of recursive algorithms

Used for:
    -Any problem that can be divided into smaller sub-problems
    -Subproblems overlap

Solutions of sub-problems are saved, avoiding the need to recalculate
-Memorization technique
'''
# In this exercise, you will implement the Fibonacci sequence, which is ubiquitous in nature. The sequence looks like this: "0, 1, 1, 2, 3, 5, 8…". You will create a recursive implementation of an algorithm that generates the sequence.

# The first numbers are 0 and 1, and the rest are the sum of the two preceding numbers.

# We can define this sequence recursively as: fib(n) = fib(n-1) + fib(n-2), with fib(0)=0 and fib(1)=1, being  the  position in the sequence.

# In the first step, you will code Fibonacci using recursion. In the second step, you will improve it by using dynamic programming, saving the solutions of the subproblems in the cache variable.

def fibonacci(n):
  # Define the base case
  if n <= 1:
    return n
  else:
    # Call recursively to fibonacci
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(6))

cache = [None]*(100)

def fibonacci(n): 
    if n <= 1:
        return n
    
    # Check if the value exists
    if not cache[n]:
        # Save the result in cache
        cache[n] = fibonacci(n-1) + fibonacci(n-2)
    
    return cache[n]
    

print(fibonacci(6))

# Great! You figured it out! You can see that dynamic programming can be very useful because the solutions of the subproblems are saved, avoiding recalculating them if needed later!

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will implement the Towers of Hanoi puzzle with a recursive algorithm. The aim of this game is to transfer all the disks from one of the three rods to another, following these rules:

# You can only move one disk at a time.
# You can only take the upper disk from one of the stacks and place it on top of another stack.
# You cannot put a larger disk on top of a smaller one.


# The algorithm shown is an implementation of this game with four disks and three rods called 'A', 'B' and 'C'. The code contains two mistakes. In fact, if you execute it, it crashes the console because it exceeds the maximum recursion depth. Can you find the bugs and fix them?

def hanoi(num_disks, from_rod, to_rod, aux_rod):
  # Correct the base case
  if num_disks >= 1:
    # Correct the calls to the hanoi function
    hanoi(num_disks - 1, from_rod, aux_rod, to_rod)
    print("Moving disk", num_disks, "from rod", from_rod,"to rod",to_rod)
    hanoi(num_disks - 1, aux_rod, to_rod, from_rod)   

num_disks = 4
source_rod = 'A'
auxiliar_rod = 'B'
target_rod = 'C'

hanoi(num_disks, source_rod, target_rod, auxiliar_rod)

##################################################################################################################################################################
##################################################################################################################################################################
'''
Searching Algorithms

1. Linear search: Searches through every element in a list one by one, O(n)
2. Binary search: Only applies to ordrered lists, O(log(n))
'''

# In this video, you learned how to implement linear search and binary search and saw the differences between them.

# In this exercise, you need to implement the binary_search() function. Can you do it?

def binary_search(ordered_list, search_value):
  first = 0
  last = len(ordered_list) - 1
  
  while first <= last:
    middle = (first + last)//2
    # Check whether the search value equals the value in the middle
    if search_value == ordered_list[middle]:
      return True
    # Check whether the search value is smaller than the value in the middle
    elif search_value < ordered_list[middle]:
      # Set last to the value of middle minus one
      last = middle - 1
    else:
      first = middle + 1
  return False
  
print(binary_search([1,5,8,9,15,20,70,72], 5))

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will implement the binary search algorithm you just learned using recursion. Recall that a recursive function refers to a function calling itself.

def binary_search_recursive(ordered_list, search_value):
  # Define the base case
  if len(ordered_list) == 0:
    return False
  else:
    middle = len(ordered_list)//2
    # Check whether the search value equals the value in the middle
    if search_value == ordered_list[middle]:
        return True
    elif search_value < ordered_list[middle]:
        # Call recursively with the left half of the list
        return binary_search_recursive(ordered_list[:middle], search_value)
    else:
        # Call recursively with the right half of the list
        return binary_search_recursive(ordered_list[middle+1:], search_value)
  
print(binary_search_recursive([1,5,8,9,15,20,70,72], 5))

##################################################################################################################################################################
'''
Binary Search Tree

-Tree where each node has 0,1,2 children
-Left subtree of a node, values less than the node itself
-Right subtree of a node, values greater than the node itself

Left and right subtree must be binary search trees

Implementations: Use the TreeNode class

Uses:
- Order lists efficiently
- Much faster at searching than arrays and linked lists
- Much faster at inserting and deleting than arrays
- Used to implement more advanced data structures: dynamic sets, lookup tables, priority queues
'''

# In the video, you learned what binary search trees (BSTs) are and how to implement their main operations.

# In this exercise, you will implement a function to insert a node into a BST.

# To test your code, you can use the following tree:

# The nodes contain titles of books, building a BST based on alphabetical order.

# This tree has been preloaded in the bst variable:

bst = CreateTree()

# You can check if the node is correctly inserted with this code:

bst.insert("Pride and Prejudice")
print(search(bst, "Pride and Prejudice"))


class BinarySearchTree:
  def __init__(self):
    self.root = None

  def insert(self, data):
    new_node = TreeNode(data)
    # Check if the BST is empty
    if self.root == None:
      self.root = new_node
      return
    else:
      current_node = self.root
      while True:
        # Check if the data to insert is smaller than the current node's data
        if data < current_node.data:
          if current_node.left_child == None:
            current_node.left_child = new_node
            return 
          else:
            current_node = current_node.left_child
        # Check if the data to insert is greater than the current node's data
        elif data > current_node.data:
          if current_node.right_child == None:
            current_node.right_child = new_node
            return
          else:
            current_node = current_node.right_child

bst = CreateTree()
bst.insert("Pride and Prejudice")
print(search(bst, "Pride and Prejudice"))

# Recall that in a binary search tree, the left subtree of a node contains only nodes with values less than the node itself, whereas the right subtree contains nodes with values greater than the node.

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will practice on a BST to find the minimum node.

class BinarySearchTree:
  def __init__(self):
    self.root = None

  def find_min(self):
    # Set current_node as the root
    current_node = self.root
    # Iterate over the nodes of the appropriate subtree
    while current_node.left_child:
      # Update the value for current_node
      current_node = current_node.left_child
    return current_node.data
  
bst = CreateTree()
print(bst.find_min())

# As you can see, we can easily find the minimum and the maximum nodes in a binary search tree because the left subtree of a node only contains nodes with values less than the node itself, and the right subtree with greater values.

##################################################################################################################################################################
'''
Depth First Search

Tree/graph traversal: Process of visiting all nodes, depth first and breadth first

DFS: In-order, pre-order, post-order
1. In-order: Left -> Current -> Right, O(n), n = # nodes, used in BST to obtain the node's values in ascending order
2. Pre-order: Current -> Left -> Right, O(n), n = # nodes, create copies of a tree, get prefix expressions
3. Post-order: Left -> Right -> Current, O(n), n = # nodes, delete binary trees, get postfix expressions

Graphs can have cycles: Need to keep track of visited vertices
1. Start at any vertex
2. Tracks current vertex to visited vertices
3. For each current node's adjacent vertex
    - If it has been visited, ignore it
    - If it hasn't been visited, recursively perform DFS
Complexity O(V+E) where V=#vertices and E=#edges

Uses a stack or queue framework
'''

# This video taught you three ways of implementing the depth first search traversal into binary trees: in-order, pre-order, and post-order.

# In the following binary search tree, you have stored the titles of some books.

# Can you apply the in-order traversal so that the titles of the books appear alphabetically ordered?

class BinarySearchTree:
  def __init__(self):
    self.root = None

  def in_order(self, current_node):
    # Check if current_node exists
    if current_node:
      # Call recursively with the appropriate half of the tree
      self.in_order(current_node.left_child)
      # Print the value of the current_node
      print(current_node.data)
      # Call recursively with the appropriate half of the tree
      self.in_order(current_node.right_child)
  
bst = CreateTree()
bst.in_order(bst.root)

# As you can see, we can apply in-order traversal to print the elements of a binary search tree in order!

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Expression trees are a kind of binary tree that represent arithmetic expressions:

import queue

class ExpressionTree:
  def __init__(self):
    self.root = None

  def pre_order(self, current_node):
    # Check if current_node exists
    if current_node:
      # Print the value of the current_node
      print(current_node.data)
      # Call pre_order recursively on the appropriate half of the tree
      self.pre_order(current_node.left_child)
      self.pre_order(current_node.right_child)
          
et = CreateExpressionTree()
et.pre_order(et.root)

# As you can see, tree traversal can have many interesting applications! In this case, pre-order traversal allowed you to obtain the prefix notation of an expression tree.

##----------------------------------------------------------------------------------------------------------------------------------------------------------------
# In this exercise, you will implement a depth first search algorithm to traverse a graph.

# Recall the steps:

# Start at any vertex
# Add the vertex to the visited vertices list
# For each current node's adjacent vertex
# If it has been visited -> ignore it
# If it hasn't been visited -> recursively perform DFS

graph = {
  '0' : ['1','2'],
  '1' : ['0', '2', '3'],
  '2' : ['0', '1', '4'],
  '3' : ['1', '4'],
  '4' : ['2', '3']
}

def dfs(visited_vertices, graph, current_vertex):
    # Check if current_vertex hasn't been visited yet
    if current_vertex not in visited_vertices:
        print(current_vertex)
        # Add current_vertex to visited_vertices
        visited_vertices.add(current_vertex)
        for adjacent_vertex in graph[current_vertex]:
            # Call recursively with the appropriate values
            dfs(visited_vertices, graph, adjacent_vertex)
            
dfs(set(), graph, '0')

# You can see that you visited all the vertices of the graph by applying the depth first search algorithm.

##################################################################################################################################################################
'''
Breadth First Search

Starts from root, then visits evry node of every level before descending
Uses a queue to search through
Is of O(n) complexity
For graphs complexity is once again O(V+E)

BFS vs DFS

BFS:
- Target is close to the starting vertex
- Applications: Web crawling, finding shortest path in unweighted graphs, finding connected locations using GPS

DFS:
- Target is far away from starting vertex
- Applications: Solving puzzles with only one solution, detecting cycles in graphs, finding shortest path in weighted graph
'''

# In this exercise, you will modify the BFS algorithm to search for a given vertex within a graph.

# To help you test your code, the following graph has been loaded using a dictionary.

graph = {
  '4' : ['6','7'],
  '6' : ['4', '7', '8'],
  '7' : ['4', '6', '9'],
  '8' : ['6', '9'],
  '9' : ['7', '8']
}

import queue

def bfs(graph, initial_vertex, search_value):
  visited_vertices = []
  bfs_queue = queue.SimpleQueue()
  visited_vertices.append(initial_vertex)
  bfs_queue.put(initial_vertex)

  while not bfs_queue.empty():
    current_vertex = bfs_queue.get()
    # Check if you found the search value
    if current_vertex == search_value:
      # Return True if you find the search value
      return True    
    for adjacent_vertex in graph[current_vertex]:
      # Check if the adjacent vertex has been visited
      if adjacent_vertex not in visited_vertices:
        visited_vertices.append(adjacent_vertex)
        bfs_queue.put(adjacent_vertex)
  # Return False if you didn't find the search value
  return False

print(bfs(graph, '4', '8'))

# As you can see, you can modify the BFS algorithm to find a given vertex by checking if we found the search value!

##################################################################################################################################################################
##################################################################################################################################################################
'''
Sorting Algorithms
- Solve how to sort an unsorted collection in ascending/descending order
- Can reduce the complexity of problems

1. Bubble sort
2. Selection sort and Insertion sort
3. Merge sort
4. Quicksort

Bubble Sort: Complexity O(n^2), best case Omega(n), average case Theta(n^2), space complexity O(1)
- First value > second value? Swap. 
- First value < second value? Do nothing.

Doesn't perform well with highly unsorted large lists
Performs well with large sorted/almost sorted lists
'''
# You have been given a program that sorts a list of numbers using the bubble sort algorithm. While testing it, you realize that the code is not correct. Could you correct the algorithm so that it works correctly?

def bubble_sort(my_list):
  list_length = len(my_list)
  # Correct the mistake
  is_sorted = False
  while not is_sorted:
    is_sorted = True
    for i in range(list_length-1):
      # Correct the mistake
      if my_list[i] > my_list[i+1]:
        my_list[i] , my_list[i+1] = my_list[i+1] , my_list[i]
        is_sorted = False
    # Correct the mistake
    list_length -= 1
  return my_list

print(bubble_sort([5, 7, 9, 1, 4, 2]))

##################################################################################################################################################################
'''
Selection Sort: Complexity O(n^2), best case Omega(n^2), average case Theta(n^2), space complexity O(1)
- Determine lowest value
- At end swap lowest value with first unordered element
- Repeat while moving forward in the list

Insertion Sort:
- Moves through the list element by element and determines if the elements before the current element are greater or less than the current element
'''
def selection_sort(my_list):
  list_length = len(my_list)
  for i in range(list_length - 1):
    # Set lowest to the element of the list located at index i
    lowest = my_list[i]
    index = i
    # Iterate again over the list starting on the next position of the i variable
    for j in range(i+1, list_length):
      # Compare whether the element of the list located at index j is smaller than lowest
      if my_list[j] < lowest:
        index = j
        lowest = my_list[j]
    my_list[i] , my_list[index] = my_list[index] , my_list[i]
  return my_list

my_list = [6, 2, 9, 7, 4, 8] 
selection_sort(my_list)
print(my_list)

##################################################################################################################################################################
'''
Merge Sort: Complexity O(n log(n)), space complexity O(n)

Divide and Conquer strategy
- Divide: Divides the problem into smaller sub-problems
- Conquer: Sub-problems are solved recursively
- Combine: Solutions of sub-problems are combined to achieve final solution

Significantly faster than bubble sort, selection/insertion sort
Suitable for sorting large lists
'''
# You have been given a program that sorts a list of numbers using the merge sort algorithm. While testing the merge_sort() function, you realize that the code is not correct. Can you correct the algorithm so that it works correctly?

def merge_sort(my_list):
    if len(my_list) > 1: 
        mid = len(my_list)//2
        left_half = my_list[:mid]
        right_half = my_list[mid:]
        
        merge_sort(left_half)
        merge_sort(right_half)
 
        i = j = k = 0
 
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
        		# Correct mistake when assigning left half
                my_list[k] = left_half[i]                
                i += 1
            else:
                # Correct mistake when assigning right half
                my_list[k] = right_half[j]
                j += 1
            k += 1
            
        while i < len(left_half):
            my_list[k] = left_half[i]
            # Correct mistake when updating pointer for left half
            i += 1
            k += 1
 
        while j < len(right_half):
            my_list[k] = right_half[j]
            # Correct mistake when updating pointer for right half
            j += 1
            k += 1

my_list = [35,22,90,4,50,20,30,40,1]
merge_sort(my_list)
print(my_list)

##################################################################################################################################################################
'''
Quicksort: Complexity O(n^2), best case Omega(n log(n)), average case Theta(n log(n)), space complexity O(n log(n))

Pivot is the firt element
Take a left and right pointer
- Move left pointer until a value greater than the pivot is found
- Move right pointer until a value lower than the pivot is found
'''
# In this exercise, you will implement the quicksort algorithm to sort a list of numbers.

# In the first step, you will implement the partition() function, which returns the index of the pivot after having processed the list of numbers so that all the elements that are to the left of the pivot are less than the pivot and all the elements that are to the right of the pivot are greater than the pivot.

# In the second step, you will implement the quicksort() function, which will call the partition() function.

def partition(my_list, first_index, last_index):
  pivot = my_list[first_index]
  left_pointer = first_index + 1
  right_pointer = last_index
 
  while True:
    # Iterate until the value pointed by left_pointer is greater than pivot or left_pointer is greater than last_index
    while my_list[left_pointer] < pivot and left_pointer < last_index:
      left_pointer += 1
    
    while my_list[right_pointer] > pivot and right_pointer >= first_index:
      right_pointer -= 1 
    if left_pointer >= right_pointer:
        break
    # Swap the values for the elements located at the left_pointer and right_pointer
    my_list[left_pointer], my_list[right_pointer] = my_list[right_pointer], my_list[left_pointer]
   
  my_list[first_index], my_list[right_pointer] = my_list[right_pointer], my_list[first_index]
  return right_pointer

def quicksort(my_list, first_index, last_index):
  if first_index < last_index:
    # Call the partition() function with the appropriate parameters
    partition_index = partition(my_list, first_index, last_index)
    # Call quicksort() on the elements to the left of the partition
    quicksort(my_list, first_index, partition_index)
    quicksort(my_list, partition_index + 1, last_index)
    
my_list = [6, 2, 9, 7] 
quicksort(my_list, 0, len(my_list) - 1)
print(my_list)