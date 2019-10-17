
class Node:
    # value <= split_value -> use left node
    # value > split_value -> use right node
    def __init__(self, feature, split_value, depth=0, label=None):
        self.label = label
        self.feature = feature
        self.split_value = split_value
        self.left_node = None
        self.right_node = None
        self.depth = depth



