import numpy as np

a = np.zeros((2,3))
arr_ones = np.ones(4)
print(a)

a=[10,2,3]
a.insert(0,0)
print(a)
a.pop(-2)
a.sort()
a.reverse()
print(a)

class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def printList(self):
        temp = self.head
        while (temp):
            print (temp.data)
            temp = temp.next

    def insert_beg(self,dataval):
        temp = Node(dataval)
        temp.next = self.head
        self.head =temp

    def at_end(self,dataval):
        newNode = Node(dataval)
        if self.head is None:
            self.head = Node
            return
        laste = self.head
        while(laste.next):
            laste = laste.next
        laste.next=newNode


llist = LinkedList()
llist.head = Node("Mon")
second = Node("Tues")
third = Node("Wed")
llist.head.next = second
second.next = third
llist.insert_beg("Sun")        #insert at the begining
llist.at_end("Thurs")           #insert at the end
llist.printList()


#BFS
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}
visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0)
    print (s, end = " ")

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'A')
print()
#DFS
# Using a Python dictionary to act as an adjacency list
graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited = [] # Set to keep track of visited nodes.

def dfs(visited, graph, node):
    if node not in visited:
        print (node, end=" ")
        visited.append(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
dfs(visited, graph, 'A')

#exchange digits on clock

test_list = [12, 1, 2, 3, 4,5,6,7,8,9,10,11]

# printing original list
print ("Original list : " + str(test_list))

# using list comprehension to left rotate by 3
k = int(len(test_list)/2)
test_list = [test_list[(i + k) % len(test_list)] for i in (test_list)]

# Printing list after left rotate
print ("List after left rotate by 3 : " + str(test_list))

# using list comprehension to right rotate by 3
# back to Original
test_list = [test_list[(i - k) % len(test_list)]
               for i, x in enumerate(test_list)]

# Printing after right rotate
print ("List after right rotate by 3(back to original) : "
                                        + str(test_list))
