import fnmatch  
import re  
from pathlib import Path  
from rllm.tools.tool_base import Tool, ToolOutput  
  
  
class FindFilesTool(Tool):  
      
    def __init__(self, max_search_results: int = 100):  
        self.max_search_results = max_search_results  
        super().__init__(  
            name="find_files",  
            description="Search for files by name pattern (supports wildcards like *.py or regex)."  
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
                        "pattern": {  
                            "type": "string",  
                            "description": 'Filename pattern (e.g., "*.py", "Dockerfile", "test_.*\\.js")'  
                        },  
                        "use_regex": {  
                            "type": "boolean",  
                            "description": "Whether to treat the pattern as a regular expression (default: false)"  
                        },  
                        "max_results": {  
                            "type": "integer",  
                            "description": "Max number of results (default: 30, max: 100)"  
                        }  
                    },  
                    "required": ["pattern"]  
                }  
            }  
        }  
      
    def forward(  
        self,  
        pattern: str,  
        repo_path: str,  
        use_regex: bool = False,  
        max_results: int = 30,  
        **kwargs  
    ) -> ToolOutput:  
        """  
        """  
        try:  
            max_results = min(max_results, self.max_search_results)  
              
            root_path = Path(repo_path)  
            if not root_path.exists() or not root_path.is_dir():  
                return ToolOutput(  
                    name=self.name,  
                    error="Repository path does not exist or is not a directory"  
                )  
              
            result = []  
              
            for file_path in root_path.rglob('*'):  
                if not file_path.is_file():  
                    continue  
                  
                try:  
                    rel_str = str(file_path.relative_to(root_path))  
                except ValueError:  
                    continue  
                  
                matched = False  
                if use_regex:  
                    try:  
                        matched = bool(re.search(pattern, rel_str))  
                    except re.error:  
                        matched = False  
                else:  
                    matched = fnmatch.fnmatch(rel_str, pattern)  
                  
                if matched:  
                    result.append(rel_str)  
                    if len(result) >= max_results:  
                        break  
              
            return ToolOutput(  
                name=self.name,  
                output={  
                    "files": result,  
                    "count": len(result),  
                    "truncated": len(result) >= max_results  
                }  
            )  
              
        except Exception as e:  
            return ToolOutput(  
                name=self.name,  
                error=f"Search failed: {str(e)}"  
            )
