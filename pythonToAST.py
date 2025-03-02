from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import json

# print("Testing execution on terminal")
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

tree = parser.parse(bytes(
    """
    def printHello():
        if bar:
            baz()
    """,
    'utf8'
))

root_node = tree.root_node
assert root_node.type == "module"   #Testing if the root node is called module
assert root_node.start_point == (1, 4)  #Testing if the module begins at line 1
assert root_node.end_point == (4, 4)    #Testing if the module ends at line 4
print(f"Root node starting point: {root_node.start_point}")
print(f"Root node end point: {root_node.end_point}")

function_node = root_node.children[0]
assert function_node.type == "function_definition"
assert function_node.child_by_field_name("name").type == "identifier"


function_name_node = function_node.children[1]
assert function_name_node.type == "identifier"
assert function_name_node.start_point == (1,8)
assert function_name_node.end_point == (1,18)
print(f"Function name node starting point: {function_name_node.start_point}")
print(f"Function name node ending point: {function_name_node.end_point}")

function_body_node = function_node.child_by_field_name("body")
print(f"Function body node length: {len(function_body_node.children)}")

if_statement_node = function_body_node.child(0)
assert if_statement_node.type == "if_statement"

function_call_node = if_statement_node.child_by_field_name("consequence").child(0).child(0) 
assert function_call_node.type == "call"

function_call_name_node = function_call_node.child_by_field_name("function")
assert function_call_name_node.type == "identifier"

function_call_args_node = function_call_node.child_by_field_name("arguments")
assert function_call_args_node.type == "argument_list"

assert str(root_node) == (
    "(module "
        "(function_definition "
            "name: (identifier) "
            "parameters: (parameters) "
            "body: (block "
                "(if_statement "
                    "condition: (identifier) "
                    "consequence: (block "
                        "(expression_statement (call "
                            "function: (identifier) "
                            "arguments: (argument_list))))))))"
)





str_root_node = str(root_node)
with(open("python_AST_extracted.json", "w", encoding="utf-8") as file):
    json.dump({"python_AST": str_root_node}, file, indent=4)


print("✅ Root node string has been saved to python_AST_extracted.json")