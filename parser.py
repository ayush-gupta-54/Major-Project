import ast

# Built-ins and common stdlib calls to exclude from the call graph
BUILTINS = {
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "list", "dict", "set", "tuple", "type",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr",
    "append", "extend", "strip", "split", "join", "format",
    "open", "read", "write", "close", "super", "staticmethod",
    "classmethod", "property", "abs", "max", "min", "sum", "any", "all",
    "int", "str", "float", "bool", "repr", "hash", "id", "iter", "next"
}


def get_call_name(node):
    """
    Extract the callee name from a Call node.

    Handles two cases:
      - ast.Name:      foo()          → "foo"
      - ast.Attribute: self.foo()     → "foo"
                       obj.bar()      → "bar"

    Returns None for anything more complex (e.g. chained calls like a.b.c())
    so we can safely skip them.
    """
    if isinstance(node.func, ast.Name):
        # Direct call: foo()
        return node.func.id

    if isinstance(node.func, ast.Attribute):
        # Method call: self.foo(), obj.bar(), data.clean()
        return node.func.attr

    # Anything else (chained calls, subscript calls, etc.) — skip
    return None


def parse_file(filepath):
    with open(filepath, "r") as f:
        source = f.read()

    tree = ast.parse(source)

    functions = []  # all top-level and nested function names
    calls = []      # (caller, callee) pairs

    # Track which function we are currently inside.
    # We use a stack so nested functions are handled correctly:
    #   def outer():
    #       def inner():   ← push "inner", pop when done
    #           foo()      ← caller is "inner", not "outer"
    call_stack = []

    class CallVisitor(ast.NodeVisitor):

        def visit_FunctionDef(self, node):
            # Register this function
            functions.append(node.name)

            # Push onto stack before visiting children
            call_stack.append(node.name)
            self.generic_visit(node)   # visit all child nodes
            call_stack.pop()           # pop after done with this function's body

        # async def functions use a different AST node — handle identically
        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Call(self, node):
            # Only record calls that happen inside a known function
            if call_stack:
                callee = get_call_name(node)
                caller = call_stack[-1]  # innermost function on the stack

                if (
                    callee is not None           # we could resolve the name
                    and callee not in BUILTINS   # not a built-in
                    and callee != caller         # ignore direct self-recursion (optional)
                ):
                    calls.append((caller, callee))

            # Continue visiting nested calls (e.g. foo(bar()))
            self.generic_visit(node)

    CallVisitor().visit(tree)

    # Deduplicate calls while preserving order
    seen = set()
    unique_calls = []
    for pair in calls:
        if pair not in seen:
            seen.add(pair)
            unique_calls.append(pair)

    return {"functions": functions, "calls": unique_calls}