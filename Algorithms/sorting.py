import random


class QuickSort:

    @staticmethod
    def quicksort(a, l, r):
        index = QuickSort.partition(a, l, r)
        if l < index - 1:
            QuickSort.quicksort(a, l, index - 1)
        if r > index:
            QuickSort.quicksort(a, index, r)

    @staticmethod
    def partition(a, l, r):
        pivot = a[(l + r) // 2]
        while l <= r:
            while a[l] < pivot: l += 1
            while a[r] > pivot: r -= 1

            if l <= r:
                a[l], a[r] = a[r], a[l]
                l += 1
                r -= 1
        return l

    @staticmethod
    def sort(a):
        QuickSort.quicksort(a, 0, len(a) - 1)


class MergeSort:

    @staticmethod
    def mergesort(a, l, r):
        if l < r:
            mid = (l + r) // 2
            MergeSort.mergesort(a, l, mid)
            MergeSort.mergesort(a, mid + 1, r)
            a[l:r + 1] = MergeSort.merge(a, l, mid, r)

    @staticmethod
    def merge(a, l, mid, r):
        merged = []
        i = l
        j = mid + 1
        while i <= mid and j <= r:
            if a[i] <= a[j]:
                merged.append(a[i])
                i += 1
            else:
                merged.append(a[j])
                j += 1

        if i == mid + 1:
            merged.extend(a[j:r + 1])
        elif j == r + 1:
            merged.extend(a[i:mid + 1])
        return merged

    @staticmethod
    def sort(a):
        MergeSort.mergesort(a, 0, len(a) - 1)


def standard_sort_qs():
    a = random.sample(range(-100, 100), 6)
    true = a.copy()
    true.sort()
    QuickSort.sort(a)
    assert (true == a)


def reverse_sort_qs():
    a = list(range(10, 0, -1))
    true = a.copy()
    true.sort()
    QuickSort.sort(a)
    assert (true == a)


def standard_sort_ms():
    a = random.sample(range(-100, 100), 6)
    true = a.copy()
    true.sort()
    MergeSort.sort(a)
    assert (true == a)


def reverse_sort_ms():
    a = list(range(10, 0, -1))
    true = a.copy()
    true.sort()
    MergeSort.sort(a)
    assert (true == a)


if __name__ == '__main__':
    standard_sort_qs()
    reverse_sort_qs()
    standard_sort_ms()
    reverse_sort_ms()
