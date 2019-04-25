class Node:

    def __init__(self, value):
        self.value = value
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0

    def insert(self, val):
        if self.head is None:
            self.head = Node(val)
            self.tail = self.head
        else:
            new_node = Node(val)
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def search(self, val):
        curr = self.head
        while curr is not None and curr.value != val:
            curr = curr.next
        if curr is None:
            return False
        return True

    def remove(self, val):
        if self.length == 0:
            return False
        if self.head.value == val:
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
        while curr.next is not None and curr.next.value != val:
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

    def push(self, val):
        if self.head is None:
            self.head = Node(val)
            self.tail = self.head
        else:
            new_node = Node(val)
            self.tail.next = new_node
            self.tail = new_node
        self.length += 1

    def pop(self):
        if self.length == 0:
            raise Exception

        temp = self.head
        self.head = self.head.next
        val = temp.value
        del temp
        return val

    def peek(self):
        if self.length == 0:
            raise Exception

        return self.head.value
