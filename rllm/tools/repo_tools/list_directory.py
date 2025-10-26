from pathlib import Path  
from rllm.tools.tool_base import Tool, ToolOutput  
  
  
class ListDirectoryTool(Tool):  
    NAME = "list_directory"
    DESCRIPTION = "List directory contents with optional depth control. Returns relative paths of files and directories."
    def __init__(self, name: str = NAME, description: str = DESCRIPTION):  
        self.max_search_results = 100  
        self.max_directory_depth = 3  
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
                            "description": 'Relative path to list (default: ".")'  
                        },  
                        "depth": {  
                            "type": "integer",  
                            "description": "Max recursion depth (default: 2, max: 3)"  
                        }  
                    },  
                    "required": ["path"]  
                }  
            }  
        }  
      
    def forward(  
        self,  
        path: str = ".",  
        depth: int = 2,  
        repo_path: str = "",  
        **kwargs  
    ) -> ToolOutput:  
        try:  
            depth = min(depth, self.max_directory_depth)  
              
            target = path  
              
            if not target.exists():  
                return ToolOutput(  
                    name=self.name,  
                    error="Path not found"  
                )  
              
            result = []  
            start_parts = Path(path).parts  
              
            for item in target.rglob('*'):  
                try:  
                    rel = item.relative_to(repo_path)  
                except ValueError:  
                    continue  
                  
                current_depth = len(rel.parts) - len(start_parts)  
                if current_depth > depth:  
                    continue  
                  
                suffix = '/' if item.is_dir() else ''  
                result.append(str(rel) + suffix)  
                  
                if len(result) >= self.max_search_results:  
                    break  
              
            result = sorted(result[:self.max_search_results])  
              
            return ToolOutput(  
                name=self.name,  
                output={  
                    "items": result,  
                    "count": len(result),  
                    "truncated": len(result) >= self.max_search_results  
                }  
            )  
              
        except Exception as e:  
            return ToolOutput(  
                name=self.name,  
                error=f"Listing failed: {str(e)}"  
            )
