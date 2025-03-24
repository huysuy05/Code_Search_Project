class ASTNode:
    def __init__(self, node_type):
        self.node_type = node_type
        self.children = []
    def add_child(self, child):
        self.children.append(child)
    def __repr__(self, level=0):
        indent = " " * level
        rep = f"{indent}{self.node_type}\n"
        for child in self.children:
            rep += child.__repr__(level + 1)
        return rep