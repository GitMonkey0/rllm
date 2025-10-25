import hydra

from datasets import load_dataset
from rllm.agents.system_prompts import SWE_GREP_SYSTEM_PROMPT
from rllm.agents.tool_agent import ToolAgent
from rllm.environments.swe_grep.swe_grep_env import SWEGrepEnvironment
from rllm.rewards.reward_fn import swe_grep_reward_fn
from rllm.trainer.agent_trainer import AgentTrainer
from rllm.tools.repo_tools import FindFilesTool, ListDirectoryTool, ParseASTTool, ReadFileTool, SearchFilesTool

@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    # train_dataset = load_dataset("parquet", data_files=config.data.train_path)["train"]

    tool_map = {"find_files": FindFilesTool, "list_directory": ListDirectoryTool, "parse_ast": ParseASTTool, "read_file": ReadFileTool, "search_files": SearchFilesTool}

    env_args = {
        "max_steps": 30,
        "tool_map": tool_map,
        "reward_fn": swe_grep_reward_fn,
    }

    agent_args = {"system_prompt": SWE_GREP_SYSTEM_PROMPT, "tool_map": tool_map, "parser_name": "qwen"}

    # Use the registry-based approach (comment out the other approach)
    trainer = AgentTrainer(
        agent_class=ToolAgent,
        env_class=SWEGrepEnvironment,
        config=config,
        # train_dataset=train_dataset,
        agent_args=agent_args,
        env_args=env_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
