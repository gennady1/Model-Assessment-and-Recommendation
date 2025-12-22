'''This class will evaluate each model using the Measuring Multitask, Language Understanding (MMLU) dataset
UPDATES:
2025/10/24 - Renamed mmlu folder variables

'''

import os
import re
import time
import pickle
from typing import List #, Dict, Any
from pathlib import Path
from datetime import timedelta
# from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

# DEBUG_LOG = True
DEBUG_LOG = False

FILTER_THINK_FROM_LLM_RESULT = True     #models that contain <think> tags
# FILTER_THINK_FROM_LLM_RESULT = False

# for debug purposes
# LLM_LIST = ['deepseek-r1:8b']
# LLM_LIST = ['phi3:3.8b', 'gemma3:4b']

LLM_LIST = [
'phi3:3.8b', 'gemma3:4b', 'codellama:7b', 'mistral:7b', 'qwen3:8b', 'deepseek-r1:8b',           # 'qwen3:8b', 'deepseek-r1:8b',      Thinking models
'mistral-nemo:12b', 'gemma3n:e4b', 'gemma3:12b-it-qat', 'phi4:14b', 'codestral:22b',
'zephyr:7b-beta-fp16', 'gemma3:27b', 'gemma3:27b-it-qat', 'llama3.3:70b']

#for debug purposes
# METRIC_LIST = ['high_school_mathematics']
# METRIC_LIST = ['elementary_mathematics', 'high_school_mathematics', 'high_school_statistics', 'college_mathematics', 'formal_logic', 'abstract_algebra']

METRIC_LIST = ['elementary_mathematics', 'high_school_mathematics', 'high_school_statistics', 'college_mathematics', 'formal_logic', 'abstract_algebra', 'high_school_computer_science', 'machine_learning', 'college_computer_science']

# Full list: ['abstract_algebra', 'all', 'anatomy', 'astronomy', 'auxiliary_train', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']


metrics_folder = "metrics"
metrics_path = os.path.join(os.getcwd(), metrics_folder)
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)

mmlu_folder = "mmlu"
mmlu_path = os.path.join(metrics_path, mmlu_folder)
if not os.path.exists(mmlu_path):
    os.makedirs(mmlu_path)

print(f" Metrics folder path: {metrics_path}")
print(f" MMLU folder path: {mmlu_path}")


class OllamaBenchmark:
    def __init__(self, model_name: str, dataset_name: str = "cais/mmlu", config_name:str ='abstract_algebra', split: str = "validation"):
        """
        Initialize the benchmarking pipeline.
        
        Args:
            model_name: Name of the Ollama model to benchmark (e.g., "gemma3:27b")
            dataset_name: Name of the Hugging Face dataset to use
            split: Dataset split to use (typically "validation")
        """
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.config_name = config_name
        self.split = split
        self.ollama_url = "http://localhost:11434"
        self.progress_file = Path(f"{mmlu_path}/mmlu_{self.model_name.replace(':', '-')}_{config_name}.pkl")

        if DEBUG_LOG:
            print(f"DEBUG:44; Progress-File: {self.progress_file}")

        #load dataset
        self.dataset = load_dataset(dataset_name, config_name, split=split)
        
        
        # Initialize progress tracking
        saved_progress = self._load_progress()
        self.correct = saved_progress['correct']
        self.completed_indices = saved_progress["completed_indices"]
        
    def _load_progress(self) -> {List[int], int}:
        """Load previous progress if it exists."""
        if self.progress_file.exists():
            with open(self.progress_file, "rb") as f:
                data = pickle.load(f)
                return data
        return {"completed_indices":[], 'correct':0}
        
    def _save_progress(self, completed_indices: List[int], correct: int):
        """Save current progress to disk."""
        with open(self.progress_file, "wb") as f:
            pickle.dump({
                "completed_indices": completed_indices,
                "correct": correct
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
            desc=f" Benchmarking [{self.config_name}]"
        )
        
        start_time = time.time()
        
        # Process each example
        for idx in range(total_examples):
            if idx in self.completed_indices:
                continue        # skip previously computed answers
                
            # Get example data
            data_sample = self.dataset[idx]
            question = data_sample["question"]
            choices = data_sample["choices"]
            correct_answer = str(data_sample["answer"])
            
            # Format prompt and get response
            prompt = self._format_prompt(question, choices)
            if DEBUG_LOG:
                print(f"DEBUG:178\n\nLLM-Query\n\tQuestion-{idx}:{question}\n\tChoices Prompt:{prompt}\n\tCorrect-Answer:{correct_answer}")
            
            response = self._query_ollama(prompt)

            if DEBUG_LOG:
                print(f"DEBUG:183\n\n  LLM-RESPONSE question-{idx}\tLabel: '{type(correct_answer)}:{correct_answer}'\tLLM-Answer: '{type(response)}:{response}'")

            if response == correct_answer:
                correct +=1
                if DEBUG_LOG:
                    print("Correct Response!!!!")
                    
            # Update progress bar
            self.completed_indices.append(idx)
            self._save_progress(self.completed_indices, correct)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            avg_time = elapsed / len(self.completed_indices)
            remaining = total_examples - len(self.completed_indices)
            eta = timedelta(seconds=int(avg_time * remaining))
            
            progress_bar.set_postfix({
                "Correct": f"{correct}/{len(self.completed_indices)}",
                "Avg (s/ex)": f"{avg_time:.2f}",
                "ETA": str(eta)
            })
            
        progress_bar.close()
        
        # Calculate and display final results; 0 <= accuracy <= 1
        accuracy = (correct / total_examples) # *100
        print(f"  Final Results:")
        print(f"  Total Examples: {total_examples}")
        print(f"  Correct Answers: {correct}")
        print(f"  Accuracy: {accuracy:.2f}%")

        return accuracy

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

    #create dataframe for capturing results 
    results = {
    'model': [],
    }

    #add metric for each model
    for metric_type in METRIC_LIST:
        results[metric_type.replace('_', ' ')] = []

    for model in model_list:
        print("\n\nEvaluating model:", model)
        results['model'].append(model)
    
        for metric_type in METRIC_LIST:
            
            benchmark = OllamaBenchmark(model_name=model, config_name=metric_type)

            # this can only be called after the OllamaBenchmark class initialized.
            # _print_dataset_answer_distribution(benchmark)
            
            # Run the benchmark
            accuracy = benchmark.run_benchmark()
            results[metric_type.replace('_', ' ')].append(accuracy)

    print(results)


    # Create DataFrame
    df = pd.DataFrame(results)

    # Write to ODS file
    print(f"Saving results to spreadsheet: {metrics_path+'/mmlu_results.ods'}")
    with pd.ExcelWriter(metrics_path+'/mmlu_results.ods', engine='odf') as writer:
        df.to_excel(writer, sheet_name='Results', index=False)

def _print_mmlu_result_logs(model_name):
    """Load and display progress if it exists."""
    progress_file = Path(f"{mmlu_path}/mmlu-{model_name.replace(':', '-')}.pkl")

    correct = 0
    completed = 0

    # Load previous progress if it exists
    if progress_file.exists():
        with open(progress_file, "rb") as f:
            data = pickle.load(f)
        correct = data['correct']
        completed = data['completed_indices'][-1]

        print(f"\tModel: {model_name}, correct: {correct}/{completed}, score: {round(correct/completed, 4) * 1}")

# Example usage
if __name__ == "__main__":

    # print results if available
    print(" Print MMLU results (if available)")
    for m in LLM_LIST:
        _print_mmlu_result_logs(m)

    # no multi threding
    _run_evaluation(model_list=LLM_LIST)

    print("\nFinished.")