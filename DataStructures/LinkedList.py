class Node:

    def __init__(self, val):
        self.val = val
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def insert(self, value):
        if self.head is None:
            self.head = Node(value)
            self.tail = self.head
        else:
            new_node = Node(value)
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def search(self, value):
        curr = self.head
        while curr is not None and curr.val != value:
            curr = curr.next
        if curr is None:
            return False
        return True

    def remove(self, value):
        if self.length == 0:
            return False
        if self.head.val == value:
            if self.length == 1:
                self.head = None
                self.tail = None
                self.length -= 1
                return True
            else:
                self.head = self.head.next
                self.length -= 1
                return True
        curr = self.head
        while curr.next is not None and curr.next.val != value:
            curr = curr.next
        if curr.next is None:
            return False
        else:
            if curr.next is self.tail:
                self.tail = curr
                self.length -= 1
                return True
            else:
                temp = curr.next
                curr.next = curr.next.next
                del temp
                self.length -= 1
                return True


class Queue:

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def push(self, value):
        if self.head is None:
            self.head = Node(value)
            self.tail = self.head
        else:
            new_node = Node(value)
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def pop(self):
        if self.length == 0:
            raise Exception

        temp = self.head
        self.head = self.head.next
        value = temp.val
        del temp
        return value

    def peek(self):
        if self.length == 0:
            raise Exception

        return self.head.val
