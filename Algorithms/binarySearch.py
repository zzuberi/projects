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


class ClosestSearch:

    @staticmethod
    def closestsearch(a, x, l, r):
        if l > r:
            if abs(x - a[r]) <= abs(x - a[l]):
                return r
            else:
                return l

        mid = (l + r) // 2
        if x < a[mid]:
            return ClosestSearch.closestsearch(a, x, l, mid - 1)
        elif x > a[mid]:
            return ClosestSearch.closestsearch(a, x, mid + 1, r)
        else:
            return mid

    @staticmethod
    def search(a, x):
        if x <= a[0]:
            return 0
        if x >= a[len(a) - 1]:
            return len(a) - 1

        return ClosestSearch.closestsearch(a, x, 0, len(a) - 1)


def standard_search():
    a = list(range(6))
    for x in a:
        result = BinarySearch.search(a, x)
        assert (x == result)
        result = ClosestSearch.search(a, x)
        assert (x == result)


def not_found_large():
    a = list(range(6))
    bin_true = -1
    result = BinarySearch.search(a, 30)
    assert (bin_true == result)
    cls_true = 5
    result = ClosestSearch.search(a, 30)
    assert (cls_true == result)


def not_found_small():
    a = list(range(6))
    bin_true = -1
    result = BinarySearch.search(a, -7)
    assert (bin_true == result)
    cls_true = 0
    result = ClosestSearch.search(a, -7)
    assert (cls_true == result)


def not_found_mid():
    a = list(range(0, 12, 2))
    bin_true = -1
    result = BinarySearch.search(a, 3)
    assert (bin_true == result)
    cls_true = 1
    result = ClosestSearch.search(a, 3)
    assert (cls_true == result)
    cls_true = 2
    result = ClosestSearch.search(a, 5)
    assert (cls_true == result)


def closest_ends():
    a = list(range(0, 12, 2))
    a.append(13)
    cls_true = 6
    result = ClosestSearch.search(a, 12)
    assert (cls_true == result)
    a.insert(0, -2)
    cls_true = 0
    result = ClosestSearch.search(a, -1)
    assert (cls_true == result)


if __name__ == '__main__':
    standard_search()
    not_found_large()
    not_found_small()
    not_found_mid()
    closest_ends()
