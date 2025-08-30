# lint_undefined.py
import ast, sys, builtins
from pathlib import Path

BUILTINS = set(dir(builtins))

class Checker(ast.NodeVisitor):
    def __init__(self):
        self.module_funcs = set()
        self.imported_names = set()
        self.class_methods = {}         # {class_name: set(method_names)}
        self.self_calls = []            # [(class_name, method_name, lineno)]
        self.bare_calls = []            # [(func_name, lineno)]
        self.class_stack = []

    # --- collection passes ---
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split(".")[0]
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.asname or alias.name
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if self.class_stack:
            cls = self.class_stack[-1]
            self.class_methods.setdefault(cls, set()).add(node.name)
        else:
            self.module_funcs.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        # treat same as FunctionDef
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name)
        self.class_methods.setdefault(node.name, set())
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_Call(self, node: ast.Call):
        # self.<method>(...)
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            method = node.func.attr
            cls = self.class_stack[-1] if self.class_stack else "<module>"
            self.self_calls.append((cls, method, node.lineno))

        # bare function call foo(...)
        elif isinstance(node.func, ast.Name):
            self.bare_calls.append((node.func.id, node.lineno))

        self.generic_visit(node)

def main(path: str):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=path)

    chk = Checker()
    chk.visit(tree)

    # Compute undefined self-method calls per class (only check against methods defined in that class)
    missing_self = []
    for cls, method, lineno in chk.self_calls:
        # If call site is not inside a class, skip
        if cls == "<module>":
            continue
        methods = chk.class_methods.get(cls, set())
        if method not in methods:
            # Note: this will flag methods inherited from mixins/parents. For this repo,
            # most helpers live in the same mixin, so this is desirable.
            missing_self.append((cls, method, lineno))

    # Compute undefined bare calls
    missing_bare = []
    for name, lineno in chk.bare_calls:
        if name in BUILTINS:  # e.g. print, len, range
            continue
        if (name not in chk.module_funcs) and (name not in chk.imported_names):
            missing_bare.append((name, lineno))

    if not missing_self and not missing_bare:
        print("✅ No obvious undefined calls found.")
        return

    if missing_self:
        print("❌ Calls to undefined self methods (by class):")
        for cls, method, ln in sorted(missing_self, key=lambda x: (x[0], x[2], x[1])):
            print(f"  {cls}.{method} at line {ln}")

    if missing_bare:
        print("\n❌ Bare function calls that are not defined/imported/built-in:")
        for name, ln in sorted(missing_bare, key=lambda x: (x[1], x[0])):
            print(f"  {name} at line {ln}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python lint_undefined.py /path/to/vision_bot.py")
        sys.exit(1)
    main(sys.argv[1])