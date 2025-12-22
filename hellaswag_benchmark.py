''' This python file runs the HELLASWAG benchmark
    UPDATE 2025-11-02 added code to track (time out) errors
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

#multithreding. Number of concurrent ollama queries
THREAD_COUNT = 3

# DEBUG_LOG = True
DEBUG_LOG = False

FILTER_THINK_FROM_LLM_RESULT = True     #models that contain <think> tags
# FILTER_THINK_FROM_LLM_RESULT = False

# LLM_LIST = [
# 'phi3:3.8b', 'gemma3:4b', 'codellama:7b', 'mistral:7b', 'qwen3:8b', 'deepseek-r1:8b',            #        Thinking models
# 'mistral-nemo:12b', 'gemma3n:e4b', 'gemma3:12b-it-qat', 'phi4:14b', 'codestral:22b', 
# 'zephyr:7b-beta-fp16', 'gemma3:27b', 'gemma3:27b-it-qat', 'llama3.3:70b']

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

class OllamaBenchmark:
    def __init__(self, model_name: str, dataset_name: str = "hellaswag", split: str = "validation"):
        """
        Initialize the benchmarking pipeline.
        
        Args:
            model_name: Name of the Ollama model to benchmark (e.g., "gemma3:27b")
            dataset_name: Name of the Hugging Face dataset to use
            split: Dataset split to use (typically "validation")
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.ollama_url = "http://localhost:11434"
        self.progress_file = Path(f"{hellaswag_path}/hellaswag-{self.model_name.replace(':', '-')}.pkl")
        
        # Load dataset
        from datasets import load_dataset
        self.dataset = load_dataset(dataset_name, split=split)
        
        # Initialize progress tracking
        saved_progress = self._load_progress()
        self.correct = saved_progress['correct']
        self.completed_indices = saved_progress["completed_indices"]
        self.missed_indices = saved_progress["missed_indices"] if "missed_indices" in saved_progress else []
        
    def _load_progress(self) -> {List[int], int}:
        """Load previous progress if it exists."""
        if self.progress_file.exists():
            with open(self.progress_file, "rb") as f:
                data = pickle.load(f)
                return data
        return {"completed_indices":[], 'correct':0, "missed_indices":[]}
        
    def _save_progress(self, completed_indices: List[int], correct: int, missed_indices: List[int]):
        """Save current progress to disk."""
        with open(self.progress_file, "wb") as f:
            pickle.dump({
                "completed_indices": completed_indices,
                "correct": correct,
                "missed_indices": missed_indices
            }, f)
            
    def _format_prompt(self, question: str, choices: List[str]) -> str:
        """Format the prompt for multiple choice questions."""
        return (
            f"Given the following question, pick the most likely choice.\n\n"
            f"Context:\n{question}\n\n"
            f"Choice list:\n"
            f"0. {choices[0]}\n"
            f"1. {choices[1]}\n"
            f"2. {choices[2]}\n"
            f"3. {choices[3]}\n\n"
            f"Respose must be a single letter that corresponds to a choice: 0, 1, 2, or 3. "
            f"DO NOT include any other information."
        )
        
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama with the given prompt."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0}
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=360
            )
            response.raise_for_status()
            answer = response.json()['response'].strip()
            
            if DEBUG_LOG:
                print("\nDEBUG:132 RESPONCE:\t", answer)

            if FILTER_THINK_FROM_LLM_RESULT:
                try:
                    digits = re.findall(r'\d', answer)[0]
                except IndexError:
                    digits = re.findall(r'\d', answer)

                if DEBUG_LOG:
                    print("\nDEBUG:141 RESPONCE-FILTERED-DIGIT\t", digits)
                return digits[-1] if digits else answer

            return answer
        except requests.RequestException as e:

            # currently the most common error is timeout, the results is counted as a miss, it tends to happen on the two thinking models (deepseek-r1:8b and the qwen3:8b). 
            print(f"Error querying Ollama: {e}")
            return None
            
    def run_benchmark(self):
        """Run the benchmarking process."""
        total_examples = len(self.dataset)
        correct = self.correct
        
        # Create progress bar
        progress_bar = tqdm(
            total=total_examples,
            initial=len(self.completed_indices),
            desc=f"Benchmarking Progress [{self.model_name}]"
        )
        
        start_time = time.time()
        
        # Process each example
        for idx in range(total_examples):
            if idx in self.completed_indices:
                continue        # skip previously computed answers
                
            # Get example data
            data_sample = self.dataset[idx]
            context = data_sample["ctx"]
            endings = data_sample["endings"]
            correct_answer = data_sample["label"]

            if DEBUG_LOG:
                print(f"\n\nLLM-Query\n\tContext:{context}\n\tEndings:{endings}\n\tCorrect-Answer:{correct_answer}")
            
            # Format prompt and get response
            prompt = self._format_prompt(context, endings)
            response = self._query_ollama(prompt)

            # print(f"\n\n  LLM-Query:\tQuestion# {idx}\tLabel:{correct_answer}\tLLM-Answer:{response}")

            if response == correct_answer:
                correct +=1

            elif response is None: # There was an error; only observed timeout error.
                self.missed_indices.append(idx)
            
            # Update progress
            self.completed_indices.append(idx)
            self._save_progress(self.completed_indices, correct, self.missed_indices)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / len(self.completed_indices)
            remaining = total_examples - len(self.completed_indices)
            eta = timedelta(seconds=int(avg_time * remaining))
            
            progress_bar.set_postfix({
                "Avg (s/ex)": f"{avg_time:.2f}",
                "ETA": str(eta)
                , "Correct" : str(correct)+'/'+str(idx+1)
                # , "Score" : f"{(correct/idx+1):.2f}"
            })
            
        progress_bar.close()
        
        # Calculate and display final results
        accuracy = (correct / total_examples) * 100
        print(f"\nFinal Results:")
        print(f"Total Examples: {total_examples}")
        print(f"Correct Answers: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")

def _print_dataset_answer_distribution(benchmark: OllamaBenchmark):
    results = {}
    
    total_examples = len(benchmark.dataset)
    for idx in range(total_examples):
        label = benchmark.dataset[idx]["label"]
        if label in results.keys():
            results[label] += 1
        else:
            results[label] = 1
            
    # Print the distribution
    for label, count in results.items():
        percentage = (count / total_examples) * 100
        print(f"Label '{label}': {count} examples ({percentage:.2f}%)")

def _run_evaluation(model_list= ['phi3:3.8b', 'gemma3:4b']):

    for m in model_list:

        print("\n\nEvaluating model:", m)
    
        benchmark = OllamaBenchmark(model_name=m)

        # this can only be called after the OllamaBenchmark class initialized.
        # _print_dataset_answer_distribution(benchmark)
        
        # Run the benchmark
        benchmark.run_benchmark()

        # If you need to aggregate or print the results:
        # print(f"Results-{m}:", results)

def _print_hellaswag_result_logs(model_name):
    """Load and display progress if it exists."""
    progress_file = Path(f"{hellaswag_path}/hellaswag-{model_name.replace(':', '-')}.pkl")
    # print(f"Loading file: {progress_file}")

    correct = 0
    completed = 0
    progress_results = []

    # Load previous progress if it exists
    if progress_file.exists():
        with open(progress_file, "rb") as f:
            data = pickle.load(f)
        correct = data['correct']
        completed = data['completed_indices'][-1]

        print(f"\tModel: {model_name}, correct: {correct}/{completed}, score: {round(correct/completed, 4) * 1}")
        # print(f"\nResults: HellaSwag Model: {model_name}, correct: {progress_results['correct']}/{progress_results['completed_indices'][-1]}, score: {round(progress_results['correct']/progress_results['completed_indices'][-1], 4) * 100}%")


# Example usage
if __name__ == "__main__":

    # print results if available
    print(" Hellaswag results:")
    for m in LLM_LIST:
        _print_hellaswag_result_logs(m)

    partitioned_llm_list = []
    for i, model in enumerate(LLM_LIST):

        #initialize the list
        if i < THREAD_COUNT:
            partitioned_llm_list.append([])

        #insert
        partitioned_llm_list[i%THREAD_COUNT].append(model)

    # now evaluate each sublist of LLMs using the _run_evaluation() function
    # Create a thread pool with as many workers as we have partitions
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        # Submit each sub‑list to the executor
        futures = [executor.submit(_run_evaluation, sublist) for sublist in partitioned_llm_list]

        # Optionally collect results as they finish
        results = []
        for future in futures:
            results.append(future.result())

    # If you need to aggregate or print the results:
    print("All LLMs evaluated:", results)