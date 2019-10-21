class Node:
    # value < split_value -> use left node
    # value >= split_value -> use right node
    # a leaf node should have a label and depth
    # a non-leaf node should have feature and split_value, and should eventually also have left_node, right_node
    def __init__(self, feature=None, split_value=None, depth=None, label=None):
        self.label = label
        self.feature = feature
        self.split_value = split_value
        self.depth = depth
        self.left_node = None
        self.right_node = None
        self.left_depth = None
        self.right_depth = None

    def set_left_node(self, node, depth):
        self.left_node = node
        self.left_depth = depth

    def set_right_node(self, node, depth):
        self.right_node = node
        self.right_depth = depth

    def is_leaf(self):
        return self.left_node is None and self.right_node is None

    def get_depth(self):
        return max(self.left_depth, self.right_depth)

    # from https://stackoverflow.com/questions/20242479/printing-a-tree-data-structure-in-python
    def __str__(self):
        if self.is_leaf():
            ret = "\t" * self.depth + repr(self.label) + "\n"
            return ret
        ret = "\t" * self.depth + repr("X{} < {}".format(self.feature, self.split_value)) + "\n"
        ret += self.left_node.__str__()
        ret += self.right_node.__str__()
        return ret

    def __repr__(self):
        return '<tree node representation>'
