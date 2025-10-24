# rllm/environments/swe_grep/swe_grep_env.py  
from rllm.environments.tools.tool_env import ToolEnvironment  
import os  
import json  

class SWEGrepEnvironment(ToolEnvironment):  
    def _execute_tool_calls(self, tool_calls):  
        repo_path = self.task.get("repo_path", "") if self.task else ""  
        
        if repo_path:  
            for tool_call in tool_calls:  
                tool_args = json.loads(tool_call["function"]["arguments"])  
                
                tool_args["repo_path"] = repo_path
                tool_call["function"]["arguments"] = json.dumps(tool_args)  
        
        return super()._execute_tool_calls(tool_calls)