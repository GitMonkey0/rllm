import fnmatch  
import re  
from pathlib import Path  
from concurrent.futures import ThreadPoolExecutor, as_completed  
from rllm.tools.tool_base import Tool, ToolOutput  
  
  
class SearchFilesTool(Tool):  
    NAME = "search_files"
    DESCRIPTION = "Search for keywords or regex in file contents across the repository."
    def __init__(self, name: str = NAME, description: str = DESCRIPTION):  
        self.max_search_results = 100  
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
                        "pattern": {  
                            "type": "string",  
                            "description": "Text to search for (literal by default, regex if use_regex=true)"  
                        },  
                        "file_pattern": {  
                            "type": "string",  
                            "description": 'File glob pattern to filter search scope (e.g. "*.py", "src/**/*.py"). Default: "*"'  
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
      
    def _find_files(self, root_path: Path, file_pattern: str):  
        if not file_pattern or file_pattern == "*":  
            all_files = []  
            for fp in root_path.rglob('*'):  
                if fp.is_file():  
                    rel = str(fp.relative_to(root_path))  
                    all_files.append(rel)  
            return all_files  
          
        patterns = [p.strip() for p in file_pattern.split(',') if p.strip()]  
        if not patterns:  
            return []  
          
        result = []  
        for fp in root_path.rglob('*'):  
            if not fp.is_file():  
                continue  
            rel = str(fp.relative_to(root_path))  
              
            for pat in patterns:  
                if fnmatch.fnmatch(rel, pat):  
                    result.append(rel)  
                    break  
          
        return result  
      
    def _search_in_file(self, root_path: Path, rel_path: str, pattern: str, use_regex: bool = False):  
        full_path = root_path / rel_path  
          
        matches = []  
        try:  
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:  
                lines = f.read().splitlines()  
        except Exception:  
            return []  

        if use_regex:  
            try:  
                compiled = re.compile(pattern, re.IGNORECASE)  
            except re.error:  
                return []  
        else:  
            compiled = re.compile(re.escape(pattern), re.IGNORECASE)  
          
        for line_num, line in enumerate(lines, 1):  
            if compiled.search(line):  
                matches.append({  
                    'file': rel_path,  
                    'line': line_num,  
                    'snippet': line.strip()  
                })  
          
        return matches  
      
    def forward(  
        self,  
        pattern: str,  
        repo_path: str,  
        file_pattern: str = "*",  
        use_regex: bool = False,  
        max_results: int = 30,  
        **kwargs  
    ) -> ToolOutput:  
        try:  
            max_results = min(max_results, self.max_search_results)  
              
            root_path = Path(repo_path)  
            if not root_path.exists() or not root_path.is_dir():  
                return ToolOutput(  
                    name=self.name,  
                    error="Repository path does not exist or is not a directory"  
                )  
              
            candidate_files = self._find_files(root_path, file_pattern)  
            results = []  
              
            with ThreadPoolExecutor(max_workers=16) as executor:  
                future_to_file = {  
                    executor.submit(self._search_in_file, root_path, rel_path, pattern, use_regex): rel_path  
                    for rel_path in candidate_files  
                }  
                  
                for future in as_completed(future_to_file):  
                    file_results = future.result()  
                    if file_results:  
                        results.extend(file_results)  
                        if len(results) >= max_results:  
                            for f in future_to_file:  
                                f.cancel()  
                            break  
              
            results = results[:max_results]  
              
            return ToolOutput(  
                name=self.name,  
                output={  
                    "matches": results,  
                    "count": len(results),  
                    "truncated": len(results) >= max_results  
                }  
            )  
              
        except Exception as e:  
            return ToolOutput(  
                name=self.name,  
                error=f"Search failed: {str(e)}"  
            )
