# rllm/environments/swe_grep/swe_grep_env.py  
from rllm.environments.tools.tool_env import ToolEnvironment  
import json  
import queue  
import threading  
  
class SWEGrepEnvironment(ToolEnvironment):  
    def _execute_tool_calls(self, tool_calls):  
        repo_path = self.task.get("repo_path", "") if self.task else ""  
          
        tool_outputs = {}  
        output_queue = queue.Queue()  
        threads = []  
  
        def execute_tool(tool_call):  
            tool_name = tool_call["function"]["name"]  
            tool_args = json.loads(tool_call["function"]["arguments"])  
              
            if repo_path:  
                tool_args["repo_path"] = repo_path  
            print(repo_path)
            tool_output = self.tools(tool_name=tool_name, **tool_args)  
            tool_output_str = tool_output.to_string()  
            output_queue.put((tool_call["id"], tool_output_str))  
  
        for tool_call in tool_calls:  
            thread = threading.Thread(target=execute_tool, args=(tool_call,))  
            threads.append(thread)  
            thread.start()  
  
        for thread in threads:  
            thread.join()  
  
        while not output_queue.empty():  
            tool_call_id, output_str = output_queue.get()  
            tool_outputs[tool_call_id] = output_str  
  
        return tool_outputs
    
    @staticmethod  
    def from_dict(env_args: dict) -> "SWEGrepEnvironment":  
        tools = env_args.pop("tools", None)  
        tool_map = env_args.pop("tool_map", None)  
        reward_fn = env_args.pop("reward_fn", None)  
        max_steps = env_args.pop("max_steps", 30)  
        
        return SWEGrepEnvironment(  
            task=env_args,   
            tools=tools,   
            tool_map=tool_map,   
            max_steps=max_steps,   
            reward_fn=reward_fn  
        )