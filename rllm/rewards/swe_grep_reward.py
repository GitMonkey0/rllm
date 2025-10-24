import re
import string
import json
from collections import Counter, defaultdict
from typing import Any, List, Dict, Tuple, Set, Optional, Union

from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput

class MULocBenchEvaluator:
    def __init__(self):
        self.metrics = {
            'file': {'acc@1': 0, 'acc@5': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'class': {'acc@1': 0, 'acc@5': 0, 'precision': 0, 'recall': 0, 'f1': 0},
            'function': {'acc@1': 0, 'acc@5': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        }
    
    def parse_location(self, loc_key: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        loc_key = loc_key.strip('()')
        parts = [p.strip().strip("'\"") for p in loc_key.split(',')]
        
        class_name = parts[0] if parts[0] != 'None' else None
        function_name = parts[1] if len(parts) > 1 and parts[1] != 'None' else None
        line_number = parts[2] if len(parts) > 2 and parts[2] != 'None' else None
        
        return class_name, function_name, line_number
    
    def extract_ground_truth(self, ground_truth: Dict) -> Dict[str, Set]:
        gt_locations = {
            'file': set(),
            'class': set(),
            'function': set()
        }
        
        for file_info in ground_truth.get('files', []):
            file_path = file_info['path']
            
            # File level
            gt_locations['file'].add(file_path)
            
            # Class and Function level
            for loc_key in file_info.get('Loc', {}):
                class_name, function_name, _ = self.parse_location(loc_key)
                
                # Class level
                if class_name:
                    gt_locations['class'].add(f"{file_path}::{class_name}")
                
                # Function level
                if function_name:
                    if class_name:
                        gt_locations['function'].add(f"{file_path}::{class_name}::{function_name}")
                    else:
                        gt_locations['function'].add(f"{file_path}::{function_name}")
        
        return gt_locations
    
    def extract_predictions(self, predictions: List[Dict]) -> Dict[str, List]:
        pred_locations = {
            'file': [],
            'class': [],
            'function': []
        }
        
        for pred in predictions:
            if not isinstance(pred, dict):
                continue
            file_path = pred.get('file')
            if not file_path:
                continue
            
            if file_path not in pred_locations['file']:
                pred_locations['file'].append(file_path)
            
            classes = pred.get('class', [])
            if classes is None:
                classes = []
            elif isinstance(classes, str):
                classes = [classes]
            
            for cls in classes:
                if cls:
                    class_loc = f"{file_path}::{cls}"
                    if class_loc not in pred_locations['class']:
                        pred_locations['class'].append(class_loc)
            
            functions = pred.get('function', [])
            if functions is None:
                functions = []
            elif isinstance(functions, str):
                functions = [functions]
            
            for func in functions:
                if func:
                    if classes:
                        for cls in classes:
                            if cls:  
                                func_loc = f"{file_path}::{cls}::{func}"
                                if func_loc not in pred_locations['function']:
                                    pred_locations['function'].append(func_loc)
                    else:
                        func_loc = f"{file_path}::{func}"
                        if func_loc not in pred_locations['function']:
                            pred_locations['function'].append(func_loc)
        
        return pred_locations
    
    def calculate_metrics(self, ground_truth: Set, predictions: List, k: int = 5) -> Dict:
        if len(ground_truth) == 0:
            return {'acc@1': 0, 'acc@5': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        top1_set = set(predictions[:1]) if len(predictions) >= 1 else set()
        topk_set = set(predictions[:k]) if len(predictions) >= k else set(predictions)
        
        acc_1 = 1.0 if ground_truth.issubset(top1_set) else 0.0
        acc_k = 1.0 if ground_truth.issubset(topk_set) else 0.0
        
        tp = len(ground_truth & topk_set)
        fp = len(topk_set - ground_truth)
        fn = len(ground_truth - topk_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'acc@1': acc_1,
            'acc@5': acc_k,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_single_issue(self, ground_truth: Dict, predictions: List[Dict], k: int = 5) -> Dict:
        gt_locations = self.extract_ground_truth(ground_truth)
        
        pred_locations = self.extract_predictions(predictions)
        
        results = {}
        for level in ['file', 'class', 'function']:
            results[level] = self.calculate_metrics(
                gt_locations[level],
                pred_locations[level],
                k=k
            )
        
        return results

class RewardSWEGrepFn:
    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        # Extract information from task_info and action
        ground_truth = json.loads(task_info["file_loc"])
        if ground_truth is None:
            return RewardOutput(reward=self.config.unk_error_reward, is_correct=False)

        try:
            predict = json.loads(action[action.find("["): action.rfind("]") + 1])
            results = MULocBenchEvaluator().evaluate_single_issue(ground_truth, predict)

            reward = (results["file"] + results["class"] + results["function"]) / 3

            return RewardOutput(reward=reward)

        except:
            return RewardOutput(reward=0.0)
