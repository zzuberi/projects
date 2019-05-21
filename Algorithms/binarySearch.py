class BinarySearch:

    @staticmethod
    def binarysearch(a, x, l, r):
        if l > r:
            return -1

        mid = (l + r) // 2
        if x < a[mid]:
            return BinarySearch.binarysearch(a, x, l, mid - 1)
        elif x > a[mid]:
            return BinarySearch.binarysearch(a, x, mid + 1, r)
        else:
            return mid

    @staticmethod
    def search(a, x):
        return BinarySearch.binarysearch(a, x, 0, len(a) - 1)


def standard_search():
    a = list(range(6))
    for x in a:
        result = BinarySearch.search(a, x)
        assert (x == result)


def not_found_large():
    a = list(range(6))
    true = -1
    result = BinarySearch.search(a, 30)
    assert (true == result)


def not_found_small():
    a = list(range(6))
    true = -1
    result = BinarySearch.search(a, -7)
    assert (true == result)


def not_found_mid():
    a = list(range(0, 6, 2))
    true = -1
    result = BinarySearch.search(a, 3)
    assert (true == result)


if __name__ == '__main__':
    standard_search()
    not_found_large()
    not_found_small()
    not_found_mid()
