''' This python file prints the HELLASWAG results only 
    UPDATES:
      2025-11-02 Removed benchmark code, added hellswag subfolder where the results are stored.
      2025-11-03 Updated the printout of skipped benchmarks to display only if there are any skipped tests.
'''

import requests
import pickle
import time
import os
import re
from tqdm import tqdm
from datetime import timedelta
from typing import List, Dict, Any
from pathlib import Path

#
from concurrent.futures import ThreadPoolExecutor

# DEBUG_LOG = True
DEBUG_LOG = False

FILTER_THINK_FROM_LLM_RESULT = True     #models that contain <think> tags
# FILTER_THINK_FROM_LLM_RESULT = False

LLM_LIST = [
'phi3:3.8b', 'gemma3:4b', 'codellama:7b', 'mistral:7b', 'qwen3:8b', 'deepseek-r1:8b',
'mistral-nemo:12b', 'gemma3n:e4b', 'gemma3:12b-it-qat', 'phi4:14b', 'codestral:22b', 
'zephyr:7b-beta-fp16', 'gemma3:27b', 'gemma3:27b-it-qat', 'llama3.3:70b']

LLM_LIST = ['qwen3:8b', 'deepseek-r1:8b']

metrics_folder = "metrics"
metrics_path = os.path.join(os.getcwd(), metrics_folder)
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

hellaswag_folder = "hellaswag"
hellaswag_path = os.path.join(metrics_path, hellaswag_folder)
if not os.path.exists(hellaswag_path):
    os.makedirs(hellaswag_path)

print(f" Metrics folder path: {metrics_path}")
print(f" Hellaswag folder path: {hellaswag_path}")

def _print_hellaswag_result_logs(model_name):
    """Load and display progress if it exists."""
    progress_file = Path(f"{hellaswag_path}/hellaswag-{model_name.replace(':', '-')}.pkl")

    correct = 0
    completed = 0
    progress_results = []

    # Load previous progress if it exists
    if progress_file.exists():
        with open(progress_file, "rb") as f:
            data = pickle.load(f)
        correct = data['correct']
        completed = data['completed_indices'][-1]
        missed_indices = data["missed_indices"] if "missed_indices" in data else []

        missed_info = ""
        if len(missed_indices) > 0:
            missed_info = f", \tNumber of skipped indecies: #{len(missed_indices)}; skipped ids: {missed_indices}. \tAdjusted score: {round(correct/(completed-len(missed_indices)), 4) * 1}"

        print(f"\tModel: {model_name}, correct: {correct}/{completed}, score: {round(correct/completed, 4) * 1} {missed_info}")


# Example usage
if __name__ == "__main__":

    # print results if available
    print(" Hellaswag results:")
    for m in LLM_LIST:
        _print_hellaswag_result_logs(m)