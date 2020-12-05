class TreeNode:

    def __init__(self, val):
        self.val = val
        self.right = None
        self.left = None

    def __str__(self):
        return str(self.val)


class BST:

    def __init__(self):
        self.root = None

    def insert(self, val):
        if self.root is None:
            self.root = TreeNode(val)
        else:
            curr = self.root
            self.__insert_node__(curr, val)

    def __insert_node__(self, curr, val):
        if val < curr.val:
            if curr.left is None:
                curr.left = TreeNode(val)
            else:
                self.__insert_node__(curr.left, val)
        else:
            if curr.right is None:
                curr.right = TreeNode(val)
            else:
                self.__insert_node__(curr.right, val)

    def contains(self, val):
        return self.__contains_node__(self.root, val)

    def __contains_node__(self, curr, val):
        if curr is None:
            return False
        elif curr.val == val:
            return True
        elif val < curr.val:
            return self.__contains_node__(curr.left, val)
        else:
            return self.__contains_node__(curr.right, val)

    def remove_node(self, val):
        return self.remove(None, self.root, val)

    def remove(self, parent, node, val):
        if node is None:
            return False
        elif val < node.val:
            return self.remove(node, node.left, val)
        elif val > node.val:
            return self.remove(node, node.right, val)

        if node.left is None and node.right is None:
            if parent is None:
                self.root = None
            elif node.val < parent.val:
                parent.left = None
            else:
                parent.right = None
        elif node.left is None:
            if parent is None:
                self.root = node.right
            elif node.val < parent.val:
                parent.left = node.right
            else:
                parent.right = node.right
            node.right = None
        elif node.right is None:
            if parent is None:
                self.root = node.left
            elif node.val < parent.val:
                parent.left = node.left
            else:
                parent.right = node.left
            node.left = None
        else:
            predecessor_parent = node
            predecessor = node.left
            while predecessor.right is not None:
                predecessor_parent = predecessor
                predecessor = predecessor.right

            predecessor_val = predecessor.val
            self.remove(predecessor_parent, predecessor, predecessor.val)
            node.val = predecessor_val
        return True

    @staticmethod
    def inorder(node, func):
        if node is None:
            return
        BST.inorder(node.left, func)
        func(node.val)
        BST.inorder(node.right, func)

    @staticmethod
    def preorder(node, func):
        if node is None:
            return
        func(node)
        BST.preorder(node.left, func)
        BST.preorder(node.right, func)

    def postorder(self, node, func):
        if node is None:
            return
        self.postorder(node.left, func)
        self.postorder(node.right, func)
        func(node)

    @staticmethod
    def is_bst(root):
        return BST.is_bst_util(root, -2 ** 32, 2 ** 32 - 1)

    @staticmethod
    def is_bst_util(node, mini, maxi):
        if node is None:
            return True

        if node.val < mini or node.val > maxi:
            return False

        return BST.is_bst_util(node.left, mini, node.val) \
               and BST.is_bst_util(node.right, node.val, maxi)

    @staticmethod
    def second_largest(root):
        if not root:
            return None
        p = None
        curr = root
        while curr.right:
            p = curr
            curr = curr.right

        if not curr.left:
            if p:
                return p.val
            else:
                return curr.val
        else:
            pred = curr.left
            while pred.right:
                pred = pred.right
            return pred.val


def passthrough(node):
    return


def createbst(l, r, tree, array):
    if l <= r:
        mid = (l + r) // 2
        tree.insert(array[mid])
        createbst(l, mid - 1, tree, array)
        createbst(mid + 1, r, tree, array)


if __name__ == "__main__":
    test = BST()
    vals = [2, 1, 4, 9, 10, 6, 5, 8, 7]
    val_preorder = []
    for z in vals:
        test.insert(z)
    test.insert(9)
    test.preorder(test.root, print)
    print()
    for z in vals:
        print(test.contains(z))
    print(test.contains(98))
    print(test.contains(-5))

    test.remove_node(9)
    test.preorder(test.root, print)

    vals = [7, 9, 4, 3, 2]
    vals2 = [1, 0, 6, 5, 8]
    test = BST()
    tree2 = BST()
    for z, val2 in zip(vals, vals2):
        test.insert(z)
        tree2.insert(val2)
    test.insert(-1)
    BST.preorder(test.root, print)
    print()
    BST.preorder(tree2.root, print)

    in_order = []
    in_order2 = []
    BST.inorder(test.root, in_order.append)
    BST.inorder(tree2.root, in_order2.append)
    print(in_order)
    print(in_order2)

    merged = [None] * (len(in_order) + len(in_order2))
    i = 0
    j = 0
    while i < len(in_order) and j < len(in_order2):
        if in_order[i] <= in_order2[j]:
            merged[i + j] = in_order[i]
            i += 1
        else:
            merged[i + j] = in_order2[j]
            j += 1
    if i == len(in_order):
        merged[i + j:] = in_order2[j:]
    elif j == len(in_order2):
        merged[i + j:] = in_order[i:]
    print(merged)
    merged_tree = BST()
    createbst(0, len(merged) - 1, merged_tree, merged)
    merged_tree.preorder(merged_tree.root, print)

    print(BST.is_bst(merged_tree.root))

    test = BST()
    vals = [5, 3, 8, 1, 4, 7, 12, 10, 9, 11]
    for z in vals:
        test.insert(z)

    print(BST.second_largest(test.root))
