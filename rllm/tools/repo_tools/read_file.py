import os  
from pathlib import Path  
from rllm.tools.tool_base import Tool, ToolOutput  
  
class ReadFileTool(Tool):  
    NAME = "read_file"
    DESCRIPTION = "Safely read a portion of a text file (with line range limit)."
    def __init__(self, name: str = NAME, description: str = DESCRIPTION):  
        self.max_lines_per_read = 500  
        super().__init__(  
            name=name,  
            description=description,  
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
                            "description": "Relative file path to read"  
                        },  
                        "start_line": {  
                            "type": "integer",  
                            "description": "Start line number (0-based, default: 0)"  
                        },  
                        "end_line": {  
                            "type": "integer",  
                            "description": "End line number (default: 200)"  
                        }  
                    },  
                    "required": ["path"]  
                }  
            }  
        }   
      
    def forward(  
        self,   
        path: str,   
        repo_path: str, 
        start_line: int = 0,   
        end_line: int = 200,   
        **kwargs  
    ) -> ToolOutput:  
        try:  
            end_line = min(end_line, start_line + self.max_lines_per_read)  
            root_path = Path(repo_path) if repo_path else Path(".")  
            full_path = root_path / path  
            
            if not full_path.is_file():  
                return ToolOutput(  
                    name=self.name,  
                    error="File not found"  
                )  
     
            with open(full_path, 'r', encoding='utf-8', errors='replace') as f:  
                lines = f.readlines()[start_line:end_line]  
              
            # content = ''.join(lines)  
            content = ''.join(f"{start_line + i}:{line}" for i, line in enumerate(lines))
              
            return ToolOutput(  
                name=self.name,  
                output={  
                    "content": content,  
                    # "start_line": start_line,  
                    # "end_line": min(end_line, start_line + len(lines)),  
                    # "total_lines": len(lines)  
                }  
            )  
              
        except Exception as e:  
            return ToolOutput(  
                name=self.name,  
                error=f"Failed to read file: {str(e)}"  
            )
