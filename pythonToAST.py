from tree_sitter import Language, Parser
import tree_sitter_python as tspython

# print("Testing execution on terminal")
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

tree = parser.parse(bytes(
    """
    def printHello():
        print("Hello")
    """,
    'utf8'
))


