''' This python file runs the semantic alignment between the interpretation and the summary.
    This class assumes the existance of ../logs/llm-output.json file (prompt inputs, interpretations, and summarizations for each model)
    UPDATE 2025-11-17 copy code from metrics_evaluate_semantic_alignment_llm_summary_aircraft_quality.py file
    UPDATE 2025-11-25 Added prompt for code generation
'''

import os
import json
import ollama
import concurrent.futures

import evaluate
import pandas as pd

# DEBUG_LOG = True
DEBUG_LOG = False

FILTER_THINKING_TAGS_FROM_LLM_RESULT = True     #models that contain <think> tags
# FILTER_THINKING_TAGS_FROM_LLM_RESULT = False

# LLM Parameters
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.05

THINKING_LLMS = ['qwen3:8b', 'deepseek-r1:8b']

#promps config
system_prompt_developer = 'You are an expert software engineer. Your task is to develop a scheduling algorithm for the airlift simulation environment'

#summarized with anything LLM / lama 3.3 / 70b
AIRLIFT_DESCRIPTION_SHORT = """
# Airlift Challenge 2.0 Simulation Environment

The **Airlift Challenge 2.0** is a discrete, graph-based simulation environment where nodes represent airports and edges represent flyable routes.
The goal is to design a scheduling algorithm that tasks aircraft to transport cargo items from source to destination airports before hard deadline.
Aircraft processing times impact scheduling timelines; higher priority aircraft are processed first.

---

## Entities and Properties:
- **airport**: Has limited processing capacity, represented by processing_capacity (airfield working MOG), and a unique airport_id.
- **aircraft**: Move between airports, carrying cargo items, with specific capacities based on the aircraft type (plane_type). Has the following attributes: aircraft_processing_time, speed, location, available_routes (from current destination), and state (WAITING, PROCESSING, MOVING, READY_FOR_TAKEOFF).
- **cargo_item**: Has the following attributes: current location, destination, weight, earliest_pickup_time, soft_deadline, and hard_deadline. Must be transported as a whole unit to their destination airport before deadline.
- **route**: Represent bidirectional flight paths between airports, with attributes like distance and mal (malfunction or unavailability property). The route becomes available when attribute mal reaches 0.

## Rules and Behavior:
- Processing aircraft: Aircraft are processed for their entire processing_time upon arrival, affecting the timeline.
- Action assignment: At each timestep, the scheduling algorithm assigns each aircraft a list of actions: unload cargo items (id list), load cargo items (id list), set destination airport ID, and assign aircraft processing priority.
- Processing cargo: Ignore cargo_items that miss their hard_deadline; if loaded, then unload.
- Priority handling: At each time step, higher priority aircraft are processed first.
- Destination airport: If the destination airport ID is set and reachable, the aircraft will take off after processing loading and unloading actions.
- The simulation environment aims to evaluate the effectiveness of scheduling algorithms in moving cargo items to their destinations while considering factors like aircraft capabilities, route availability, and deadline constraints.
"""

TAKS_ASSIGNMENT_TEMPLATE = """
def assign_aircraft_actions(self):
        '''
        This method needs to assigns list of actions for each aircraft. This method must return a dictionary of action items for each aircraft:

        actions = {}

        actions[aircraft_id] = {
                        "cargo_to_unload": [],          # list of cargo_id to unload from aircraft
                        "cargo_to_load": [],            # list of cargo_id to load from current airport
                        "destination": NOAIRPORT_ID,    # next airport destination id
                        "priority": 1                   # priority aircraft
                        }
                

            aircraft states: MOVING → WAITING → PROCESSING → READY_FOR_TAKEOFF 
        '''

        actions = {a: None for a in self.aircraft}
        
        # assign each aircraft actions: unload or load cargo items, set destination, and aircraft's priority
        for aid, a in self.aircraft.items():

            #initialize action list for each aircraft agent
            actions[aid] = {
                            "cargo_to_unload": [],
                            "cargo_to_load": [],
                            "destination": NOAIRPORT_ID,
                            "priority": 1}


            #task assignment code goes here


        return actions
"""

EXAMPLE_AIRLIFT_OBJECTS = '''
Example cargo item 'c_0':
'c_0' = CargoObservation(id=0, location=8, destination=5, weight=2, earliest_pickup_time=66, is_available=False, soft_deadline=606, hard_deadline=1146)

Example aircraft agent 'a_0' :
'a_0' = {'available_routes': [3, 1, 2],
        'cargo_at_current_airport': [],
        'cargo_onboard': [],
        'current_airport': 4,
        'current_weight': 0,
        'destination': 0,
        'max_weight': 5,
        'next_action': {'cargo_to_load': [],
                        'cargo_to_unload': [],
                        'destination': 0,
                        'priority': 1} 
        'plane_type': 0,
        'state': PlaneState.READY_FOR_TAKEOFF}
}'''

MOG_FOLDER = "mog"
MOG_FOLDER_PATH = os.path.join(os.getcwd(), MOG_FOLDER)
if not os.path.exists(MOG_FOLDER_PATH):
    os.makedirs(MOG_FOLDER_PATH)

LLM_RESULTS_FOLDER = "llm-results"
LLM_RESULTS_FOLDER_PATH = os.path.join(MOG_FOLDER_PATH, LLM_RESULTS_FOLDER)
if not os.path.exists(LLM_RESULTS_FOLDER_PATH):
    os.makedirs(LLM_RESULTS_FOLDER_PATH)

# name of the file for storing proposed algorithms
FILEPATH_ALGORITHM_DESIGNS = LLM_RESULTS_FOLDER_PATH+f'/algorithm_design.json'
FILEPATH_ALG_DESIGN_INFORMATION_RETANTION = LLM_RESULTS_FOLDER_PATH+f'/information_retantion_algorithm_design_similarity_scores.ods'
FILEPATH_ALGORITHM_CODE = LLM_RESULTS_FOLDER_PATH+f'/algorithm_code.json'
FILEPATH_SCHED_CODE_INFORMATION_RETANTION = LLM_RESULTS_FOLDER_PATH+f'/information_retantion_code_similarity_scores.ods'

print(f" MOG folder path: {MOG_FOLDER_PATH}")
print(f" LLM results folder path: {LLM_RESULTS_FOLDER_PATH}")

# --------------------------------------------------------
# These are helper functions for hugging face evaluate 
# method calls (in paralle)
# --------------------------------------------------------

def compute_bleu(predictions, reference):
    return evaluate.load("bleu").compute(predictions=predictions, references=reference)

def compute_rouge(predictions, reference):
    try:
        r = evaluate.load("rouge").compute(predictions=predictions, references=[reference])
        return {'rouge1': float(r['rouge1']), 'rouge2' : float(r['rouge2']), 'rougeL' : float(r['rougeL']), 'rougeLsum' : float(r['rougeLsum'])}
    except Exception as e:
        print(f"\n\n\t** Exception-ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0 }

def compute_meteor(predictions, reference):       # <-- tell Python to use the global variable

    return float(evaluate.load('meteor').compute(predictions=predictions, references=reference)['meteor'])
def compute_bertscore(predictions, reference):
    try:
        r = evaluate.load('bertscore', model_type="roberta-large").compute(lang="en", predictions=predictions, references=reference)
        return {'precision': r['precision'][0], 'recall': r['recall'][0], 'f1': r['f1'][0]}
    except Exception as e:
        print(f"\n\t** Exception-BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def generate_algorithm_recommendations_with_llm(model, combined_sim_results_summary):
    '''This method generates a scheduling algorithm design for the airlift environment'''
    print(f"\n  Generating results for: {model}")

    #prepare template for storing results
    result = {'summarize_input_prompt':'', 'combined_summary_response':'', 'algorithm_design_input_prompt':'', 'algorithm_design_response':''}

    # Step 1. Summarize results to few salient feature
    sys_prompt_summarization = 'You are an expert data statistician. Your task is to summarized the aggregate results airlift simulation environment.'
    prompt_input_task = '''
    *Task*
    Your task is to identify consice, salient features from the summarized results found in the __Data__ section.

    **Context**
    These salient features will be used to design a scheduling algorithm for the airlift scheduling simulation environment.

    **Data**
    '''+combined_sim_results_summary

    response = ollama.chat(model=model, messages=[ {
                'role': 'system',
                'content': sys_prompt_summarization,     #1. set model context 
            }, {
                'role': 'user',
                'content': prompt_input_task
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            # 'top_p': LLM_TOP_P,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )

    #record the results
    result['summarize_input_prompt'] = prompt_input_task
    result['combined_summary_response'] = response['message']['content']

    
    # Step 2. get algorith recomendatin
    system_prompt_alg_design = 'You are an expert software engineer and data scientist. Your task is to recommend a scheduling algorithm for a simulation environment'     #1. set model context
    prompt_input_alg_design = f"""{{
    *Task*
    Your task is to identify a good scheduling algorithm for pickup and delivery of cargo for the Airlift Challenge 2.0 simulation environment.

    **Objective**
    The primary objective of the scheduling algorithm is to task aircraft to pickup and deliver cargo items before hard deadlines. The secondary objective is to minimize the distance traveled.
    The goal of the simulation environment is to design effective strategies that move cargo_items to their destination airport before the deadlines.

    **Context**
    These salient features will be used to design a scheduling algorithm for the airlift scheduling simulation environment.

    **Data**
    **Simulation environment description**
    {AIRLIFT_DESCRIPTION_SHORT}

    **Operational insights from analysis of simulation data**
    {result['combined_summary_response']}
    }}"""  #2. get algorith recomendatin

    result['algorithm_design_input_prompt'] = prompt_input_alg_design

    # **Additional requirements**
    # {alg_requirements_description_json}

    response = ollama.chat(model=model, messages=[ {
                'role': 'system',
                'content': system_prompt_alg_design,     #1. set model context 
            }, {
                'role': 'user',
                'content': prompt_input_alg_design
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            # 'top_p': LLM_TOP_P,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )

    #save the result
    result['algorithm_design_response'] = response['message']['content']


    return result

def create_algorithm_design_recommendations(llm_results):
    ''' This method loads combined LLM summaries for each model and creates a model design.
    '''

    # 1. aggregate llm summaries for each simulated variable
    combined_model_summary_results = {}
    for sim_var in llm_results.keys():

        for model in llm_results[sim_var]['combined']:
            # 1. get llm summary
            llm_summary = llm_results[sim_var]['combined'][model]['summary']

            #for thinking LLMs. Remove thinking tags Simple method that checks if there are open and closed <think> tags (both required)
            if model in THINKING_LLMS and FILTER_THINKING_TAGS_FROM_LLM_RESULT:
                index1 = llm_summary.find("<think>")
                index2 = llm_summary.find("</think>")

                if index1>0 and index2>index1:
                    llm_summary = llm_summary[:index1] + llm_summary[index2+8:].strip()
            
            # add results to dictionary
            if model not in combined_model_summary_results:
                combined_model_summary_results[model] = f"**Summary of simulation results for simulated variable: {sim_var}**\n\n{llm_summary}"
            else:
                combined_model_summary_results[model] += f"\n\n**Summary of simulation results for simulated variable: {sim_var}**\n\n{llm_summary}"

    # 2. call LLM to create design
    algorithm_design_dic = {}
    for m, combined_summary in combined_model_summary_results.items():
        algorithm_design_dic[m] = generate_algorithm_recommendations_with_llm(model=m, combined_sim_results_summary=combined_summary)
        
    # 3. Save results to a file
    print(f"Saving results to spreadsheet: {FILEPATH_ALGORITHM_DESIGNS}")
    with open(FILEPATH_ALGORITHM_DESIGNS, 'w', encoding="utf-8") as f:                            # Save as json
        json.dump(algorithm_design_dic, f)
    
    return algorithm_design_dic

def load_llm_output_logs_and_analyze_semantic_comparison_of_summarized_text():
    '''
    This method loads the llm generated results (JSON format) and evaluates the results metrics provided by evaluate python library (huggingface)
    '''
    llm_results = {}

    #load llm generated data from logs
    logs_folder = 'logs'
    file_path = os.getcwd() + '/' + logs_folder + '/llm-output.json'
    # file_path = os.getcwd() + '/' + logs_folder + '/llm-output-small.json'
    with open(file_path, 'r', encoding="utf-8") as f:
        llm_results = json.load(f)
    
    print(f" Loaded llm_results, file: {file_path}")

    algorithm_design_dic = create_algorithm_design_recommendations(llm_results=llm_results)

    return algorithm_design_dic

# Convert dictionaries to DataFrames with expanded metrics
def expand_dataframe_scores(df, title):
    # calculate the length teh title
    var = list(df.values())[0]
    if type(var) is dict or type(var) is list:
        title_len = len(var)
    else:
        title_len = 1

    df = pd.DataFrame(list(df.items()), columns=['Model', 'Score'])
    df = df.join(df['Score'].apply(pd.Series))                  # Split nested dict into columns
    df = df.drop(columns=['Score'])                             # Drop the original 'Score' column to match expected columns
    df.loc[-1] = [title] + [''] * title_len                     # Add title row
    df = df.sort_index()
    return df
def calculate_task_to_output_infomration_retantion(algorithm_design_recommendations_dic):

    similarity_types = ['BERTscore', 'BLEU', 'ROUGE', 'METEOR']
    output_df = {}

    for st in similarity_types:

        # 1. calculate information retation for the summarization step
        print(f"\n  Calculating information retation for the summarization using {st}")
        information_retantion_summarized_results = {}
        for model, result in algorithm_design_recommendations_dic.items():
            print(f"\tModel: {model} \t {len(result['summarize_input_prompt'])} \t{len(result['combined_summary_response'])}")
            if st == 'BERTscore':
                information_retantion_summarized_results[model] = compute_bertscore([result['combined_summary_response']], [result['summarize_input_prompt']])
            elif st == 'BLEU':
                information_retantion_summarized_results[model] = compute_bleu([result['combined_summary_response']], [result['summarize_input_prompt']])
            elif st == 'ROUGE':
                information_retantion_summarized_results[model] = compute_rouge([result['combined_summary_response']], [result['summarize_input_prompt']])
            elif st == 'METEOR':
                information_retantion_summarized_results[model] = compute_meteor([result['combined_summary_response']], [result['summarize_input_prompt']])
            
            # print(f"\tModel: {model} \t {len(result['summarize_input_prompt'])} \t{len(result['combined_summary_response'])}\tscore = {information_retantion_summarized_results[model]}")

        # 2. calculate information retation for algorithm recommendation
        print(f"\n  Calculating information retation for algorithm recommendations using {st}")
        information_retantion_recommendation_results = {}
        for model, result in algorithm_design_recommendations_dic.items():
            print(f"\tModel: {model} \t {len(result['algorithm_design_input_prompt'])} \t{len(result['algorithm_design_response'])}")
            if st == 'BERTscore':
                information_retantion_recommendation_results[model] = compute_bertscore([result['algorithm_design_input_prompt']], [result['algorithm_design_response']])
            elif st == 'BLEU':
                information_retantion_recommendation_results[model] = compute_bleu([result['algorithm_design_input_prompt']], [result['algorithm_design_response']])
            elif st == 'ROUGE':
                information_retantion_recommendation_results[model] = compute_rouge([result['algorithm_design_input_prompt']], [result['algorithm_design_response']])
            elif st == 'METEOR':
                information_retantion_recommendation_results[model] = compute_meteor([result['algorithm_design_input_prompt']], [result['algorithm_design_response']])

            # print(f"\tModel: {model} \t {len(result['algorithm_design_input_prompt'])} \t{len(result['algorithm_design_response'])}\tscore = {information_retantion_recommendation_results[model]}")

        # print(f"\n\ninformation_retantion_summarization_results = {information_retantion_summarized_results}")
        # print(f"information_retantion_recommendation_results = {information_retantion_recommendation_results}")

        # Process summarization results
        df_summarization = expand_dataframe_scores(information_retantion_summarized_results, "Summarization of results")

        # Process recommendation results
        df_recommendation = expand_dataframe_scores(information_retantion_recommendation_results, "Algorithm recommendations")

        # Insert blank column between sections
        blank_col = pd.DataFrame({'': [''] * 2})  # Blank column
        combined_df = pd.concat([df_summarization, blank_col, df_recommendation], ignore_index=True)

        #record results
        output_df[st] = combined_df

    # save the results as open document sheet (ODS)
    with pd.ExcelWriter(FILEPATH_ALG_DESIGN_INFORMATION_RETANTION, engine='odf') as writer:
        for st, df in output_df.items():
            df.to_excel(writer, sheet_name=st, index=False)

def analyse_simulation_results_and_generate_candidate_scheduling_algorithm():
    print("\nMethod: analyse_simulation_results_and_generate_candidate_scheduling_algorithm() - This method generates candidate scheduling algorithms for the airlift simulation environment")

    #1. check if the LLM algorithm recommendations exist. if not create them
    if not os.path.exists(FILEPATH_ALGORITHM_DESIGNS):
        print(f" Algorithm design file NOT found ({FILEPATH_ALGORITHM_DESIGNS}). Generating algorithm recommendations\t")
        load_llm_output_logs_and_analyze_semantic_comparison_of_summarized_text()


    #2. load the results
    algorithm_design_recommendations_dic = {}
    with open(FILEPATH_ALGORITHM_DESIGNS, 'r', encoding="utf-8") as f:
        algorithm_design_recommendations_dic = json.load(f)
    print(f" Loaded algorithm designs file: {FILEPATH_ALGORITHM_DESIGNS}")

    #3. calculate task to ouput information retantion
    calculate_task_to_output_infomration_retantion(algorithm_design_recommendations_dic)

def create_python_code_that_assigns_actions_to_aircraft(algorithm_design_recommendations_dic) -> dict:

    candidate_assignment_algorithms = {}

    # relvant dictionary keys: 'algorithm_design_input_prompt':'', 'algorithm_design_response'
    for model, values in algorithm_design_recommendations_dic.items():

        prompt_tasker = f"""{{
        **TASK**
        Implement the recommended scheduling algorithm for tasking aircraft in the airlift simulation environment. 


        **INSTRUCTIONS**
        Use the `RECOMMENDED ALGORITHM DESIGN`, `AIRLIFT CHALLENGE 2.0`, and `EXAMPLE AIRLIFT OBJECTS` descriptions to implment a task assignment method `assign_aircraft_actions()` in Python for the Airlift simulation environment. Implement your code using this template: 
        `{TAKS_ASSIGNMENT_TEMPLATE}`
        

        **RECOMMENDED ALGORITHM DESIGN**
        {values['algorithm_design_response']}


        **EXAMPLE AIRLIFT OBJECTS**
        {EXAMPLE_AIRLIFT_OBJECTS}


        **ASSUME THESE HELPER METHODS ALREADY IMPLEMENTED**
        get_cargo_items(self, state):       # returns a list of active cargo items
        prioratize_cargo_items(self):       # Simple helper method that sorts cargo_items based on their hard_deadline, and then best on earliest pickup time
        get_path_cost(self, start: Airport, end: Airport, plane: PlaneType):        # Returns a summed cost of edges from start airport to end airport (uses shortest path algorithm).
        calculate_cargo_delivery_paths(self):       #Calculates a shortest path for each cargo item to cargo destination


        **AIRLIFT CHALLENGE 2.0**
        {AIRLIFT_DESCRIPTION_SHORT}
        }}"""

        print(f"\tModel: {model}, lenght: {len(prompt_tasker)}")

        # call LLM
        response = ollama.chat(model=model, messages=[ {
                    'role': 'system',
                    'content': system_prompt_developer,     #1. set model context 
                }, {
                    'role': 'user',
                    'content': prompt_tasker
                },
            ],
            options = {
                'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
            }
        )

        #save the result
        candidate_assignment_algorithms[model] = {'prompt_code_implementation': prompt_tasker, 'code': response['message']['content']}

    #return the results
    return candidate_assignment_algorithms


def calculate_code_prompt_to_output_infomration_retantion(scheduling_code_dic):

    similarity_types = ['BERTscore', 'BLEU', 'ROUGE', 'METEOR']
    output_df = {}

    for st in similarity_types:

        # 1. calculate information retation for the summarization step
        print(f"\n  Calculating information retation for the generated code. Similarity benchmark = {st}")
        information_retantion_summarized_results = {}
        for model, result in scheduling_code_dic.items():
            print(f"\tModel: {model} \t {len(result['prompt_code_implementation'])} \t{len(result['code'])}")
            if st == 'BERTscore':
                information_retantion_summarized_results[model] = compute_bertscore([result['prompt_code_implementation']], [result['code']])
            elif st == 'BLEU':
                information_retantion_summarized_results[model] = compute_bleu([result['prompt_code_implementation']], [result['code']])
            elif st == 'ROUGE':
                information_retantion_summarized_results[model] = compute_rouge([result['prompt_code_implementation']], [result['code']])
            elif st == 'METEOR':
                information_retantion_summarized_results[model] = compute_meteor([result['prompt_code_implementation']], [result['code']])
            
            # print(f"\tModel: {model} \t {len(result['prompt_code_implementation'])} \t{len(result['code'])}\tscore = {information_retantion_summarized_results[model]}")

        # print(f"\n\ninformation_retantion_summarization_results = {information_retantion_summarized_results}")

        # Process summarization results
        df_summarization = expand_dataframe_scores(information_retantion_summarized_results, "Information retention for generated code")

        #record results
        output_df[st] = df_summarization

    # save the results as open document sheet (ODS)
    with pd.ExcelWriter(FILEPATH_SCHED_CODE_INFORMATION_RETANTION, engine='odf') as writer:
        for st, df in output_df.items():
            df.to_excel(writer, sheet_name=st, index=False)

    return output_df

def generate_python_task_assignment_code_from_algorithm_recommendations():
    '''This method generates a scheduling algorithm design for the airlift environment'''
    print("\nMethod: generate_python_task_assignment_code_from_algorithm_recommendations() - This method generates a scheduling algorithm design for the airlift simulation environment")

    # 1. Try loading existing dictionary with generated algorithm code
    airlift_scheduling_code = {}
    if not os.path.exists(FILEPATH_ALGORITHM_CODE):

        # 1. Get algorithm designs - Check if can load an existing algorithm design file, if not create it
        algorithm_design_recommendations_dic = {}
        if not os.path.exists(FILEPATH_ALGORITHM_DESIGNS):
            print(f" File with Scheduling Algorithm Designs NOT found ({FILEPATH_ALGORITHM_DESIGNS}). Generating new file (this step will take some time) ...")
            algorithm_design_recommendations_dic = load_llm_output_logs_and_analyze_semantic_comparison_of_summarized_text()

        else:
            # Load existing dictionary with results
            with open(FILEPATH_ALGORITHM_DESIGNS, 'r', encoding="utf-8") as f:
                algorithm_design_recommendations_dic = json.load(f)

            model_code_snippets = [model for model, val in algorithm_design_recommendations_dic.items()]
            print(f" Loaded algorithm design recommendations file: {FILEPATH_ALGORITHM_DESIGNS}: Models: {model_code_snippets}")

        # Use LLM to implement python code (tasks aircraft to pick up and deliver cargo items...)
        airlift_scheduling_code = create_python_code_that_assigns_actions_to_aircraft(algorithm_design_recommendations_dic)

        # Save results to a file
        print(f"Saving results to file: {FILEPATH_ALGORITHM_CODE}")
        with open(FILEPATH_ALGORITHM_CODE, 'w', encoding="utf-8") as f:                            # Save as json
            json.dump(airlift_scheduling_code, f)
    else:

        #load existing results
        with open(FILEPATH_ALGORITHM_CODE, 'r', encoding="utf-8") as f:
            airlift_scheduling_code = json.load(f)

        print(f" Loaded code dictionary file: {FILEPATH_ALGORITHM_DESIGNS}")

    #Check for missing code (code lenght == 0)
    missing_model_code = []
    model_code_snippets = ""
    for model, val in airlift_scheduling_code.items():

        code_length = len(val['code'])

        if code_length == 0:
            missing_model_code.append(model)
        model_code_snippets += f"\n\t\t{model}\t\tcode length: {code_length}"
    
    print("\n  Loaded file with code implementations:", model_code_snippets)

    # try to fix missing code
    if len(missing_model_code) > 0:
        print(f"\n\tFound missing code for the following models: {missing_model_code}")
        for m in missing_model_code:
            print(f"\t\tGenerating new code for: {m}")
        #generate code for that model
            response = ollama.chat(model=m, messages=[ {
                        'role': 'system',
                        'content': system_prompt_developer,     #1. set model context 
                    }, {
                        'role': 'user',
                        'content': val['prompt_code_implementation']
                    },
                ],
                options = {
                    'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
                }
            )

        #save the result
        airlift_scheduling_code[m] = {'prompt_code_implementation': airlift_scheduling_code[m]['prompt_code_implementation'], 'code': response['message']['content']}
        
        #save results to file
        with open(FILEPATH_ALGORITHM_CODE, 'w', encoding="utf-8") as f:                            # Save as json
            json.dump(airlift_scheduling_code, f)
        print(f"\tSaved the updated code to file: {FILEPATH_ALGORITHM_CODE}")

    return airlift_scheduling_code


def calculate_semantic_alignment_between_prompt_code_design_and_generated_code():

    #load/generate dictionary with scheduling algorithm designs
    airlift_scheduling_code = generate_python_task_assignment_code_from_algorithm_recommendations()

    # 1. Try load existing file first
    algorithm_code_semantic_alignment_score = {}
    if os.path.exists(FILEPATH_SCHED_CODE_INFORMATION_RETANTION):

        #Evaluate the information retantion between the input prompt and the generated output prompt, check if results already exist
        algorithm_code_semantic_alignment_score = pd.read_excel(FILEPATH_SCHED_CODE_INFORMATION_RETANTION, engine='odf')
        print(f"\tLoaded existing result, file: {FILEPATH_SCHED_CODE_INFORMATION_RETANTION}")

    else:
        # calculate the semantic retantion and save
        algorithm_code_semantic_alignment_score = calculate_code_prompt_to_output_infomration_retantion(airlift_scheduling_code)
       
    # Print results
    print(f" Loaded algorithm design recommendations file: {FILEPATH_SCHED_CODE_INFORMATION_RETANTION}")
    for i, (key, value) in enumerate(list(airlift_scheduling_code.items())):
        print(f"\t{i+1}\t{key}\t{value['code'][:10].strip()}")


# test_metrics()
# analyse_simulation_results_and_generate_candidate_scheduling_algorithm()
# generate_python_task_assignment_code_from_algorithm_recommendations()
calculate_semantic_alignment_between_prompt_code_design_and_generated_code()


# print("*****Debug")
# result = {}
# result['algorithm_design_input_prompt'] = "Some sample text. Hello world."
# result['algorithm_design_response'] = "Just another sample text."
# print("A")
# r = evaluate.load('bertscore', model_type="roberta-large").compute(lang="en", predictions=[result['algorithm_design_input_prompt']], references=[result['algorithm_design_response']])
# print("B")
# print(f"\tTesting similarity\t {len(result['algorithm_design_input_prompt'])} \t{len(result['algorithm_design_response'])}")
# print("Done...")