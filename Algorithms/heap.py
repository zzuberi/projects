class Heap:

    def __init__(self):
        self.heap = []

    def push(self, a):
        self.heap.append(a)
        self.siftup(len(self.heap) - 1)

    def siftup(self, index):
        if index == 0:
            return

        if index % 2 == 1:
            parent = (index - 1) // 2
        else:
            parent = (index - 2) // 2

        if self.heap[parent] > self.heap[index]:
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]

            self.siftup(parent)

    def pop(self):
        val, self.heap[0] = self.heap[0], self.heap[-1]

        del self.heap[-1]
        self.siftdown(0)

    def siftdown(self, index):
        if index == len(self.heap) - 1:
            return

        for x in range(1, 3):
            child = 2 * index + x
            if child < len(self.heap) and self.heap[index] > self.heap[child]:
                self.heap[child], self.heap[index] = self.heap[index], self.heap[child]

                self.siftdown(child)

    def heapify(self, a):
        index = len(a) - 1
        if index % 2 == 1:
            parent = (index - 1) // 2
        else:
            parent = (index - 2) // 2

        for x in range(parent, -1, -1):
            self.siftdown(x)
