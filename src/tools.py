"""Small, isolated Python calculation tool for model-generated snippets.

The tool permits arithmetic/collection code plus narrow imports and attributes
from itertools, math and fractions, and a short allowlist of local collection
methods. Each call runs in a fresh process with CPU, address-space, wall-time
and output limits. It is a calculator sandbox, not a general untrusted-code
hosting service.
"""
import json
import subprocess
import sys
import textwrap


_WORKER = r'''
import ast, itertools, json, math, resource, sys
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
payload = json.load(sys.stdin)
code = payload.get("code", "")
if len(code) > 12000:
    raise SystemExit("code too long")
tree = ast.parse(code, mode="exec")
allowed_nodes = (ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.For, ast.If,
    ast.While, ast.Break, ast.Continue, ast.Pass, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.Call, ast.Subscript, ast.Slice, ast.ListComp,
    ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension, ast.Add,
    ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub,
    ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.MatMult,
    ast.JoinedStr, ast.FormattedValue, ast.Import, ast.ImportFrom, ast.alias,
    ast.Attribute, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
    ast.keyword, ast.Lambda)
allowed_names = {"print", "range", "len", "sum", "min", "max", "abs", "int",
    "float", "str", "round", "sorted", "enumerate", "zip", "all", "any", "list",
    "tuple", "set", "dict", "map", "filter", "pow", "divmod", "combinations",
    "permutations", "product", "factorial", "gcd", "lcm", "comb", "Fraction",
    "itertools", "math", "fractions"}
safe_object_methods = {"append", "extend", "get", "items", "keys", "values",
    "count", "index", "split", "join", "replace", "strip", "lower", "upper", "sort"}
safe_string_methods = {"count", "index", "split", "join", "replace", "strip", "lower", "upper"}
module_attributes = {"itertools": {"combinations", "permutations", "product"},
                     "math": {"factorial", "gcd", "lcm", "comb", "sqrt", "ceil", "floor",
                              "log", "log10", "sin", "cos", "tan", "asin", "acos", "atan",
                              "degrees", "radians", "pi", "e", "tau", "inf"},
                     "fractions": {"Fraction"}}
defined_functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
if any(name.startswith("__") for name in defined_functions):
    raise SystemExit("disallowed function name")
for node in ast.walk(tree):
    if not isinstance(node, allowed_nodes):
        raise SystemExit("disallowed syntax: " + type(node).__name__)
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        raise SystemExit("disallowed name")
    if isinstance(node, ast.Import) and any(alias.name not in module_attributes for alias in node.names):
        raise SystemExit("only itertools, math, and fractions may be imported")
    if isinstance(node, ast.ImportFrom) and (node.module not in module_attributes or node.level != 0 or
            any(alias.name not in module_attributes[node.module] for alias in node.names)):
        raise SystemExit("disallowed import")
    if isinstance(node, ast.Attribute):
        allowed_module_attribute = (isinstance(node.value, ast.Name) and
            node.value.id in module_attributes and node.attr in module_attributes[node.value.id])
        allowed_local_method = isinstance(node.value, ast.Name) and node.attr in safe_object_methods
        allowed_literal_string_method = (isinstance(node.value, ast.Constant) and
            isinstance(node.value.value, str) and node.attr in safe_string_methods)
        if not (allowed_module_attribute or allowed_local_method or allowed_literal_string_method):
            raise SystemExit("disallowed attribute")
    if isinstance(node, ast.Call):
        allowed_named_call = isinstance(node.func, ast.Name) and node.func.id in allowed_names | defined_functions
        allowed_module_call = (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and node.func.value.id in module_attributes and
            node.func.attr in module_attributes[node.func.value.id])
        allowed_local_method_call = (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Name) and node.func.attr in safe_object_methods)
        allowed_literal_string_call = (isinstance(node.func, ast.Attribute) and
            isinstance(node.func.value, ast.Constant) and isinstance(node.func.value.value, str) and
            node.func.attr in safe_string_methods)
        if not (allowed_named_call or allowed_module_call or allowed_local_method_call or
                allowed_literal_string_call):
            raise SystemExit("disallowed function call")
safe = {name: getattr(itertools, name) for name in ("combinations", "permutations", "product")}
safe.update({name: getattr(math, name) for name in module_attributes["math"]})
# An explicit builtin allowlist avoids exposing eval/open/import/compile.
import builtins
safe.update({name: getattr(builtins, name) for name in allowed_names if hasattr(builtins, name)})
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in {"itertools", "math", "fractions"}:
        raise ImportError("module is not available in the calculator")
    return builtins.__import__(name, globals, locals, fromlist, 0)
safe["__import__"] = safe_import
safe["combinations"] = itertools.combinations
safe["permutations"] = itertools.permutations
safe["product"] = itertools.product
exec(compile(tree, "<tool>", "exec"), {"__builtins__": safe}, {})
'''


def run_python(code: str, timeout_seconds: float = 3.0, max_output_chars: int = 2000):
    """Execute restricted calculation code and return a serializable tool result."""
    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "output": "empty code"}
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", _WORKER], input=json.dumps({"code": code}),
                              text=True, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired:
        return {"ok": False, "output": "execution timed out"}
    output = (proc.stdout + proc.stderr).strip()
    if len(output) > max_output_chars:
        output = output[:max_output_chars] + "\n[output truncated]"
    return {"ok": proc.returncode == 0, "output": output or ("completed without output" if proc.returncode == 0 else "tool failed")}


def extract_python_action(text: str):
    """Return the first fenced python action, if present."""
    import re
    match = re.search(r"```(?:python|py)\s*\n(.*?)```", text, flags=re.I | re.S)
    return textwrap.dedent(match.group(1)).strip() if match else None
