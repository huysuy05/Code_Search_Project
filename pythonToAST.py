from tree_sitter import Language, Parser
import tree_sitter_python as tspython
from ASTNode import *

# print("Testing execution on terminal")
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

def pythonToAST(code):
    """Returns a Tree object by parsing the tree and getting the root node"""
    tree = parser.parse(bytes(code, encoding="utf-8"))
    objAST = tree.root_node
    assert tree.root_node.type == "module"  # Ensures that the root node of Python code is "module"
    return objAST


# def nodeToDict(node):
#     """Returns a dictionary format of the AST"""
#     return {
#         "type": node.type,
#         "start_point": node.start_point,
#         "end_point": node.end_point,
#         "children": [nodeToDict(child) for child in node.children]
#     }


def ASTSummarization(objAST):  # ---> The AST of the code snippet
    # tree = parser.parse(bytes(objAST, encoding="utf-8"))
    """Perform a recursive function for an in-order traversal starting from the AST's root node"""
    def findBodyNode(node):
        if node.type == 'block':
            return node
        for child in node.children:
            res = findBodyNode(child)
            if res:
                return res
        return None

    try:
        body_node = findBodyNode(objAST)
        listSeqs = []
        for child in body_node.children:
            listSeqs.append(child.type)
        return listSeqs
    except AttributeError:
        print("Could not find the root body node")
        return []


def ASTSumRepresentation(listSeqs):
    """Returns a tree representation of ASTSum after extracting from the original AST"""
    root = None
    repr = []
    for node_type in listSeqs:
        non_terminal_node = ASTNode(node_type)
        if not root:
            root = non_terminal_node
            repr.append(root)
        else:
            parent_node = repr[0]
            parent_node.add_child(non_terminal_node)
            repr = [non_terminal_node]
        

    return root




def test():
    code = """
def example_function():
    clk = obj.clk
    rst = obj.rst
    for u in obj._units:
        _tryConnect(clk, u, "clk")
        _tryConnect(rst, u, "rst_n")
        _tryConnect(rst, u, 'rst')
"""
    objAST = pythonToAST(code)
    listSeqs = ASTSummarization(objAST)
    print(listSeqs)
    print(ASTSumRepresentation(listSeqs))
test()