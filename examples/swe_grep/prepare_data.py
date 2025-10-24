import json    
import os    
from rllm.data.dataset import DatasetRegistry    
    
    
def prepare_open_swe_grep_data(repo_base_path, train_size=None, test_size=None):    
    """    
    Loading Open-SWE-Grep dataset and registering it with the DatasetRegistry.    
        
    Args:    
        repo_base_path: Base path for repositories    
        train_size: Maximum number of training examples to load    
        test_size: Maximum number of test examples to load    
        
    Returns:    
        Dataset instance  
    """    
        
    def process_split(json_path, max_size, split_name):    
        """Process a data split with optional size limit"""    
        with open(json_path, 'r', encoding='utf-8') as f:    
            raw_data = json.load(f)    
            
        if max_size is not None:    
            raw_data = raw_data[:max_size]    
            
        processed = []    
        for idx, example in enumerate(raw_data):    
            repo_suffix = f"{example['organization']}_{example['repo_name']}_{example['base_commit']}"    
            repo_path = os.path.join(repo_base_path, repo_suffix)    
              
            file_loc = example.get("file_loc", [])  
            file_loc_str = json.dumps(file_loc) if file_loc else "[]"  
                
            data = {    
                "data_source": "open_swe_grep",    
                "prompt": [{"role": "user", "content": example["question"]}],    
                "ability": "swe",    
                "reward_model": {"style": "rule", "ground_truth": ""},    
                "extra_info": {    
                    "split": split_name,    
                    "index": idx,    
                    "organization": example["organization"],    
                    "repo_name": example["repo_name"],    
                    "base_commit": example["base_commit"],    
                    "repo_path": repo_path,    
                    "file_loc": file_loc_str
                }    
            }    
            processed.append(data)    
            
        print(f"Processed {len(processed)} examples from {split_name}")    
        return processed    
        
    print("Loading Open-SWE-Grep dataset...")    
        
    train_json_path = "data/train/v1/train_994.json"    
    train_processed = process_split(train_json_path, train_size, "train")    
        
    print(train_processed[0])  
    train_dataset = DatasetRegistry.register_dataset("open_swe_grep", train_processed, "train")    
        
    return train_dataset.get_data()    
    
    
if __name__ == "__main__":    
    repo_base = "data/train/v1/repos"    
    train_dataset = prepare_open_swe_grep_data(repo_base)    
    print(f"Train dataset first example: {train_dataset[0]}")    
    print(f"Repo path: {train_dataset[0]['extra_info']['repo_path']}")