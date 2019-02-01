'''
ICP 2 CS 5590 UMKC
Kenton Hanifl
2/1/2019
'''

class Stack:
    '''
    Implements a stack using a list
    '''
    def __init__(self):
        self.stack = list()
        
    def pop(self):
        try:
            ret = self.stack[-1]
            del self.stack[-1]
            return ret
        except:
            return "Nothing in the stack"

    def push(self, obj):
        self.stack.append(obj)

    def top(self):
        try:
            ret = self.stack[-1]
            return ret
        except:
            return "Nothing in the stack"

    def getContents(self):
        return self.stack


class Queue:
    '''
    Implements a queue using a list
    '''
    def __init__(self):
        self.queue = list()
        
    def enqueue(self, obj):
        self.queue.append(obj)

    def dequeue(self):
        try:
            ret = self.queue[0]
            del self.queue[0]
            return ret
        except:
            return "Nothing in the queue"

    def front(self):
        try:
            ret = self.queue[0]
            return ret
        except:
            return "Nothing in the queue"

    def getContents(self):
        return self.queue


def changeCase(string):
    '''
    String: An alphanumeric string
    
    Returns: A string with the cases reversed from the original string
    '''
    retString = ""
    
    for char in string:
        if char.islower():
            retString += char.upper()
        elif char.isupper():
            retString += char.lower()
        else: # if neither upper nor lower case
            retString += char
            
    return retString
    

def plants():
    '''
    Asks the user for a number of plants
    Then asks the user for the heights separated by spaces of these plants

    Prints the avg height of all of these plants
    '''
    NumPlants = int(input("Enter a number of plants: "))
    Heights = input("Enter the heights of the plants: ")
    HeightsSep = Heights.split(" ")
    Total = 0
    for Height in HeightsSep:
        Total += int(Height)
    Avg = Total/NumPlants
    formatAvg = "{0:.3f}".format(Avg)
    print(formatAvg)

def testStack():
    '''
    Tests the stack class found in this document
    '''
    s = Stack()
    print("checking new stack.top() and trying to pop")
    s.top()
    s.pop()
    print("pushing 1 and 2 to the stack")
    s.push(1)
    s.push(2)
    print("check the top of the stack")
    print(s.top())
    print("popping twice")
    print(s.pop())
    print(s.pop())
    print("trying to pop again from the empty stack")
    s.pop()

def testQueue():
    '''
    Tests the queue class found in this document
    '''
    q = Queue()
    print("checking new queue.front() and trying to dequeue")
    q.front()
    q.dequeue()
    print("pushing 1 and 2 to the queue")
    q.enqueue(1)
    q.enqueue(2)
    print("check the front of the queue")
    print(q.front())
    print("dequeueing twice")
    print(q.dequeue())
    print(q.dequeue())
    print("trying to dequeue from an empty queue")
    q.dequeue()

def main():
    plants()
    stackLoop()
    queueLoop()
    inStr = input("Enter a string to reverse the case of: ")
    print(changeCase(inStr))

def stackLoop():
    print("This is a stack interface.")
    stack = Stack()
    cont = True
    while(cont):
        choice = input("\n1: Print stack.top()\n2: Call stack.push()\n3: Call stack.pop()\n4: Print contents of Stack\n5: Quit")
        if choice[0] == "1":
            print(stack.top())
        elif choice[0] == "2":
            push = input("Please input a value to push")
            stack.push(push)
        elif choice[0] == "3":
            print(stack.pop())
        elif choice[0] == "4":
            print(stack.getContents())
        else:
            cont = False

def queueLoop():
    print("This is a queue interface.")
    queue = Queue()
    cont = True
    while(cont):
        choice = input("\n1: Print queue.front()\n2: Call queue.enqueue()\n3: Call queue.dequeue()\n4: Print contents of Queue\n5: Quit")
        if choice[0] == "1":
            print(queue.front())
        elif choice[0] == "2":
            enqueue = input("Please input a value to enqueue")
            queue.enqueue(enqueue)
        elif choice[0] == "3":
            print(queue.dequeue())
        elif choice[0] == "4":
            print(queue.getContents())
        else:
            cont = False

main()



