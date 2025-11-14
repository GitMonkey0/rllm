import ast  
from pathlib import Path  
from rllm.tools.tool_base import Tool, ToolOutput  
from typing import Iterator, Tuple

Line = int
Name = str
Sig = str
Kind = str
Scope = str

def iter_all_symbols(content: str) -> Iterator[Tuple[Line, Name, Sig, Kind, Scope]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return

    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = f"def {node.name}(" + ",".join(arg.arg for arg in node.args.args) + ")"
            scope = ""
            p = parents.get(node)
            while p:
                if isinstance(p, ast.ClassDef):
                    scope = p.name
                    break
                p = parents.get(p)
            yield node.lineno, node.name, sig, "function", scope

        elif isinstance(node, ast.ClassDef):
            bases = ",".join(ast.unparse(b) for b in node.bases) if node.bases else ""
            sig = f"class {node.name}" + (f"({bases})" if bases else "")
            yield node.lineno, node.name, sig, "class", ""
  
class ParseASTTool(Tool):  
    NAME = "parse_ast"
    DESCRIPTION = "Parse a Python file and return AST index."
    def __init__(self, name: str = NAME, description: str = DESCRIPTION):  
        self.max_ast_results = 50  
        super().__init__(  
            name=name,  
            description=description
        )  
      
    @property  
    def json(self):  
        return {  
            "type": "function",  
            "function": {  
                "name": self.name,  
                "description": self.description,  
                "parameters": {  
                    "type": "object",  
                    "properties": {  
                        "path": {  
                            "type": "string",  
                            "description": "Relative .py path"  
                        }
                    },  
                    "required": ["path"]  
                }  
            }  
        }  
      
    def _root_mod(self, name: str) -> str:  
        return name.split(".", 1)[0]   
      
    def forward(  
        self,   
        path: str,   
        repo_path: str,  
        **kwargs  
    ) -> ToolOutput:  
        try:  
            root_path = Path(repo_path) if repo_path else Path(".")  
            full_path = root_path / path  
            
            if full_path.suffix.lower() != ".py":  
                return ToolOutput(  
                    name=self.name,  
                    error="Only .py files supported"  
                )  
              
            try:  
                content = full_path.read_text(encoding="utf-8", errors="ignore")  
            except Exception as e:  
                return ToolOutput(  
                    name=self.name,  
                    error=f"Read error: {str(e)}"  
                )  
              
            try:  
                tree = ast.parse(content)  
            except SyntaxError as e:  
                return ToolOutput(  
                    name=self.name,  
                    error=f"Syntax error in Python file: {str(e)}"  
                )  
              
            imports, globals_, functions, classes = [], [], [], []  
              
            for node in tree.body:  
                if isinstance(node, ast.Import):  
                    imports.extend(self._root_mod(alias.name) for alias in node.names)  
                elif isinstance(node, ast.ImportFrom) and node.module:  
                    imports.append(self._root_mod(node.module))  
                elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):  
                    var_name = node.targets[0].id  
                    globals_.append([node.lineno, var_name])  
              
            imports = sorted(set(imports))  
              
            class_methods = {}  
            for line_no, name, sig, kind, scope in iter_all_symbols(content):  
                if kind == "function":  
                    if scope:  
                        class_methods.setdefault(scope, []).append([line_no, name])  
                    else:  
                        functions.append([line_no, name, sig])  
                elif kind == "class":  
                    bases = []  
                    for cls_node in tree.body:  
                        if isinstance(cls_node, ast.ClassDef) and cls_node.name == name:  
                            try:  
                                bases = [ast.unparse(b) for b in cls_node.bases]  
                            except Exception:  
                                bases = [ast.dump(b) for b in cls_node.bases]  
                            break  
                    classes.append([line_no, name, bases, class_methods.get(name, [])])  
              
            result = {  
                "imports": imports[:self.max_ast_results],  
                "globals": globals_[:self.max_ast_results],  
                "functions": functions[:self.max_ast_results],  
                "classes": classes[:self.max_ast_results],  
                "truncated": {  
                    "imports": len(imports) > self.max_ast_results,  
                    "globals": len(globals_) > self.max_ast_results,  
                    "functions": len(functions) > self.max_ast_results,  
                    "classes": len(classes) > self.max_ast_results  
                }  
            }  
              
            return ToolOutput(  
                name=self.name,  
                output=result  
            )  
              
        except Exception as e:  
            return ToolOutput(  
                name=self.name,  
                error=f"Parse failed: {str(e)}"  
            )
