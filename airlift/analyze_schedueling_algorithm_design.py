import ollama
import json

#UPDATES:
# 20250703 migration to Markdown from JSON

#interactive use case:
# https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/prompt_format.md

# ------------------ LLM Configuration ------------------

# ---- do not work. ---- 
# LLM_MODEL = 'deepseek-coder:latest'       # 0.78 GB   # useful for checking if the code runs... 
# LLM_MODEL = 'llama3.2:latest'	            # 2.0 GB    # sometimes fails


# ---- smaller models, testing ----

# LLM_MODEL = 'phi3:latest'	                # 2.3 GB
# LLM_MODEL = 'wizardcoder:latest'	        # 3.8 GB
# LLM_MODEL = 'codellama:7b-python'	        # 3.8 GB
# LLM_MODEL = 'codellama:latest'            # 3.8 GB
# LLM_MODEL = 'codellama:7b'                # 3.8 GB
# LLM_MODEL = 'llama2-uncensored:latest'    # 3.8 GB
# LLM_MODEL = 'llama2:latest'	              # 3.8 GB
# LLM_MODEL = 'llama3.1:8b'                 # 4.9 GB        # meh performance


# ---- good models (size to performance) ----
LLM_MODEL = 'mistral-nemo:12b'              # 7.1 GB
# LLM_MODEL = 'gemma2:27b'                  # 15 GB


# ---- bigger models ----
# LLM_MODEL = 'zephyr:141b'                 # 79 GB
# LLM_MODEL = 'mistral-large:123b'          # 69 GB
# LLM_MODEL = 'codellama:34b-instruct-fp16' # 67 GB
# LLM_MODEL = 'codestral:22b-v0.1-f16'      # 44 GB
# LLM_MODEL = 'llama3.3:latest'             # 42 GB
# LLM_MODEL = 'llama3.1:70b'                # 39 GB

# ---- big models ----
# LLM_MODEL = 'phi3:14b-medium-128k-instruct-f16'   # 27 GB
# LLM_MODEL = 'granite-code:34b'                    # 19 GB
# LLM_MODEL = 'zephyr:7b-beta-fp16'                 # 14 GB

# ---- bad performance ----
# LLM_MODEL = 'starcoder2:7b'     # 4 GB

# LLM_TEMPERATURE = 1.5 # very creative; 
LLM_TEMPERATURE = .3 # Somewhat conservative (good for coding and correct syntax)
# LLM_TEMPERATURE = .01 # Very conservative (mostly repetable)
# LLM_TEMPERATURE = 0  # Repetable

# LLM_LOGIT_BIAS = 1.5 # 1.5: very creative
LLM_TOP_P = .3

PRINT_LLM_INPUT  = False
PRINT_LLM_OUTPUT = True

NUMBER_OF_SUMMARY_ITERATIONS = 2    # uses LLM to summarize the analysis results n times; selects the best summary; same is used for the summarization of the simulation environment.

print(f"\n\n-----------------------------------------------------------\nLLM Configuration:\n   Model: {LLM_MODEL}\n   Temperature: {LLM_TEMPERATURE}\n   Top_P: {LLM_TOP_P}\n   Print LLM Input: {PRINT_LLM_INPUT}\n   Print LLM Output: {PRINT_LLM_OUTPUT}" )


#list of files containing summarization
files_name_list = ['LLM_Recomendation1_Aircraft_Quantity.txt', 'LLM_Recomendation2_wMOG_capacity.txt', 'LLM_Recomendation3_Processing_time.txt', 'LLM_Recomendation4_Poisson_lambda.txt', 'LLM_Recomendation5_Max_Duration.txt']

def strip_json_markup(json_text):
    #strip non JSON data.
    start_index = json_text.find('{')
    end_index = json_text.rfind('}')
    if start_index != -1 and end_index!= -1:
        json_text = json_text[start_index:end_index]
    json_text = json_text +"}"
    return json_text

# reads the file
def read_analysis_file(filename):
    with open(filename, 'r') as f:
        analysis_doc = f.read()
    return analysis_doc

# pretty print json formated data
def pretty_print_json(json_obj):
    print(json.dumps(json_obj, indent=4))

# #print model details
# print(pretty_print_json(ollama.show(LLM_MODEL)))

# this will load the summarized analysis
file_prefix = "./pdf/"
list_summarized_analysis = []
for file in files_name_list:
    file =  file_prefix + file
    print(f"\n\tReading file: '{file}'")
    analysis_doc = read_analysis_file(file)
    list_summarized_analysis.append(analysis_doc)

analzed_sim_data = ""
for result in list_summarized_analysis:
    analzed_sim_data += (result+",\n\n")

analzed_sim_data = analzed_sim_data


input_to_llm_analysis_data = f"""
**Task**
I need summarizaiton of simulation results. The results are in JSON format.

**Specific instructions**
1. Succinctly summarize the analysis.
2. Identify insightful information, bottlenecks is often a cause of disruptions.
3. Provide top 2 key salient recommendations that are useful for the design of schedueling algorithm design.
4. Format the output as JSON with simple hierarchy.

**Simulation results**
{analzed_sim_data}
"""

if PRINT_LLM_INPUT:
  print("\nLLM INPUT:\n", input_to_llm_analysis_data)

# TODO strip out recommendations from the summary, compare against original, ask if the summary is representative of the original text, if yes, then compare which one is better recommendation.

summary_analysis_lst = []

print("\nGenerating summary recommendations:")
for i in range(NUMBER_OF_SUMMARY_ITERATIONS):

    print(f"\tIteration-{i}")

    response = ollama.chat(model=LLM_MODEL, messages=[
        {
        'role': 'system',
        'content': 'You are an expert data scientist',     #1. set model context 
        },
        {
        'content': input_to_llm_analysis_data,
        'role': 'user',
        },
    ],
    options = {
        # 'logit_bias': LLM_LOGIT_BIAS,
        'top_p': LLM_TOP_P,
        'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
    }
    )
    sim_data_analysis_summary = response['message']['content']

    #strip non JSON data. Usually the output contains additioanl non JSON text, the code below removes it.
    start_index = sim_data_analysis_summary.find('{')
    end_index = sim_data_analysis_summary.rfind('}')
    if start_index != -1 and end_index!= -1:
        sim_data_analysis_summary = sim_data_analysis_summary[start_index:end_index]
    sim_data_analysis_summary = sim_data_analysis_summary +"}"
    summary_analysis_lst.append(sim_data_analysis_summary)



best_analysis_summary = ""
print("\n\n----Comparing summaries:")
for i, summary in enumerate(summary_analysis_lst):

    #strip out recommendations block
    start_index = summary.find('"recommendations":')
    end_index = summary.rfind(']')
    if start_index != -1 and end_index!= -1:
        summary = "{ " + summary[start_index:end_index+1] + "\n}"
        
    response = ollama.chat(model=LLM_MODEL, messages=[ {
              'role': 'user',
              'content': 'You are an expert data scientist',     #1. set model context 
          }, {
              'content': f"""This is the verbose analysis results:
              {input_to_llm_analysis_data}


              Here are two summaries of the above results. Please printout the best summarized recommendations:
                            
              
              Summary-1:
              {summary}
              

              Summary-2:
              {best_analysis_summary}        
              
              From the best summary, output only the "recommendations" and it's sub elements in the JSON format.
              Do not include JSON markers.
              """,
              'role': 'user',
          }, 
      ], options = {
          'temperature': LLM_TEMPERATURE # 0.1: very conservative (good for coding and correct syntax)
      }
    )
    #parse string into JSON object
    best_analysis_summary = response['message']['content']

print(f"\n<--- Best Simulation Results Summary (n={NUMBER_OF_SUMMARY_ITERATIONS}):\n{best_analysis_summary}\n-->\n\n")


# LLM templates for the algorithm design:
sim_description_verbose = """
# Description: Airlift simulation environment

This document describes the entities, their attributes, and rules in an Airlift Challenge 2.0 simulation environment. The goal of the simulation environment is to design effective strategies that move cargo_items to their destination airport before the deadlines. This is a discrete graph-based simulation environment, where nodes represent airports and edges represent flyable routes.
The scheduling algorithm assigns each aircraft a list of actions: unloading cargo items, loading cargo items, setting destination airport ID, and assigning processing priority.

## Entities and Properties

The system models several key components:
- **Airport**: A location (node) with limited processing capacity.
- **Aircraft**: Provides airlift transportation between airports, carrying cargo items. Each aircraft has specific type properties affecting their capabilities.
- **Cargo Item**: Each cargo_item must be transported (as a whole unit) to the destination airport before deadlines.
- **Route**: Represents a bidirectional flight route between two airports.

### Airport
- **Description:** Airports are locations (nodes) in the simulation graph. Each airport has limited working capacity (processing_capacity), dictating how many aircraft can be serviced simultaneously.
  - `airport_id`: Unique identifier for an airport.
  - `processing_capacity`: The maximum number of aircraft that can be processed at this airport concurrently. Also known as airfield working MOG.

### Aircraft
- **Description:** Aircraft move between airports and transfer cargo_items. They have specific capabilities based on their type (plane_type). Represented as nodes in a graph.
  - `state`: Current operational state: 'WAITING', 'PROCESSING', 'MOVING' or 'READY_FOR_TAKEOFF'.
    - Attributes:
      - `aircraft_processing_time`: The time it takes to process an aircraft upon arrival. *Note: This is defined by the aircraft's plane_type.*
      - `speed`: How quickly the aircraft moves between adjacent airports. *Note: This is defined by the aircraft's plane_type.*
  - `location`: Location airport_id, ID of the current airport (0 if en route).
  - `available_routes`: List of airports IDs that are directly reachable from the current location.

### Cargo Item
- **Description:**  Each has cargo_item has: current location (location airport_id), destination (airport_id), weight, earliest_pickup_time, soft_deadline, and hard_deadline.
  - `location`: Location airport_id of the airfield where the cargo_item is at. The airport_id = 0 when cargo_item is loaded on the aircraft.
  - `destination`: Location airport_id, ID of the destination airport.
  - `weight`: The cargo's weight. Must fit within an aircraft's current_weight capacity. 
  - `earliest_pickup_time`: Cargo becomes available for pickup at or after this time step. This is also the earliest time it can be loaded onto an aircraft.
  - `soft_deadline`: Target delivery time; if missed, a small penalty is incurred. 
  - `hard_deadline`: Strict delivery target; if missed, significant penalty is incurred. Avoid loading cargo with passed hard_deadline.

### Route
- **Description:** Represents a flight path between two airports. Routes are bidirectional edges in the graph.
  - `distance`: The distance of this edge (in some unit, probably steps or distance units).
  - `mal`: Each route has a 'malfunction' property, or unavailability. Route is available if 'mal' = 0. A value greater than 0 implies that route is not fliable. Ex: If 'mal' is 5, the route will become available when 'mal' reachies 0 after 5 simulation time steps.

## Rules and Behavior

**Key Principles:**

1.  ### Aircraft Processing ###
    Upon arrival, an aircraft is processed for its entire 'processing_time', affecting the timeline.

2.  **Action Assignment**: The scheduling algorithm assigns each aricraft a list of actions: unload cargo_items, load cargo_items, set destination airport_id, and assigns processing priority for the aircraft.

3.  **Priority Handling**: At each time-step, the simulation procces aircraft based on a priority level. Higher assigned priority aircraft will be processed before lower ones at each time step of the simulation environment.

4. **Destination Airport**: If the destination airport_id is set and the airport is reachable, the aircraft will take off after processing loading and unloading actions.
"""

#summarized with anything LLM / lama 3.3 / 70b
good_summary_example = """
{
# Summarized description: Airlift simulation environment
The Airlift Challenge 2.0 simulation environment is a discrete graph-based model where nodes represent airports and edges represent flyable routes. The goal is to design effective strategies that move cargo items to their destination airports before hard deadlines.


## Entities and Properties:

- **airport**: Have limited processing capacity, represented by processing_capacity (airfield working MOG), and a unique airport_id.
- **aircraft**: Move between airports, carrying cargo items, with specific capabilities based on their type (plane_type). They have attributes such as state, aircraft_processing_time, speed, location, and available_routes.
- **cargo_item**: Have properties like location, destination, weight, earliest_pickup_time, soft_deadline, and hard_deadline. Must be transported as a whole unit to their destination airport before deadlines.
- **route**: Represent bidirectional flight paths between airports, with attributes like distance and mal (malfunction or unavailability property). The route becomes available when attribute mal reaches 0.


## Rules and Behavior:

- Processing aircraft: Aircraft are processed for their entire processing_time upon arrival, affecting the timeline.
- Action assignment: The scheduling algorithm assigns each aircraft a list of actions: unload cargo items, load cargo items, set destination airport ID, and assign aircraft processing priority.
- Priority handling: Aircraft are processed based on their assigned priority level at each time step.
- Destination airport: If the destination airport ID is set and reachable, the aircraft will take off after processing loading and unloading actions.
- The simulation environment aims to evaluate the effectiveness of scheduling algorithms in moving cargo items to their destinations while considering factors like aircraft capabilities, route availability, and deadline constraints.
"""

input_to_llm_alg_desc = f"""I have a verbose description of the simulation environment in Markdown format:
 
{sim_description_verbose}
 
Commands:
 Please succinctly summarize the description.
 Please only include salient features and insightful information.
 Please format the output as JSON using simple hierarchy.
 Do not include JSON markers.

 Here is an example of good summary:
 {good_summary_example}
 """


#
list_summarized_alg_descriptions = []
for i in range(NUMBER_OF_SUMMARY_ITERATIONS):
    response = ollama.chat(model=LLM_MODEL, messages=[
        {
        'role': 'system',
        'content': 'You are an expert software engineer',     #1. set model context 
        },
        {
        'content': input_to_llm_alg_desc,
        'role': 'user',
        },
    ],
    options = {
        # 'logit_bias': LLM_LOGIT_BIAS,
        'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
    }
    )
    sim_description_summary = response['message']['content']

    list_summarized_alg_descriptions.append(sim_description_summary)


#lets choose the best sim description:
best_sim_env_summary = ""
print("\n\n----Comparing summaries:")
for i, summary in enumerate(list_summarized_alg_descriptions):

    #strip out recommendations block
    start_index = summary.find('"recommendations":')
    end_index = summary.rfind(']')
    if start_index != -1 and end_index!= -1:
        summary = "{ " + summary[start_index:end_index+1] + "\n}"
        
    response = ollama.chat(model=LLM_MODEL, messages=[ {
              'role': 'user',
              'content': 'You are an expert software engineer',     #1. set model context 
          }, {
              'content': f"""This is the verbose description of the simulation environment:
              {sim_description_verbose}


              Here are two summaries of the above description. Please printout the most useful summary for designing a schedueling algorithm:
                            
              
              Summary-1:
              {summary}
              

              Summary-2:
              {best_sim_env_summary}        
              
              Ouput in JSON format only.
              Do not include JSON markers.
              """,
              'role': 'user',
          }, 
      ], options = {
          'temperature': LLM_TEMPERATURE # 0.1: very conservative (good for coding and correct syntax)
      }
    )
    #parse string into JSON object
    best_sim_env_summary = response['message']['content']

print(f"\n<--- Best description:\n{best_sim_env_summary}\n-->\n\n")

#strip non JSON data.
best_analysis_summary = strip_json_markup(best_analysis_summary)
best_sim_env_summary = strip_json_markup(best_sim_env_summary)


#note these 
alg_requirements_description_json = """
  "algorithm_design": {
    "additional-ino": [
      "info" : "A 'Score' of a simulation run represents the overall effectivness of the scheduling algorithm used.",
      "info" : "The 'Score' is the sum of all penalties incurred by the scheduler.",
      "info" : "The smaller the 'Score' is, the better the solution",
      "info" : "There are two deadlines, 'soft-deadline' and 'hard-deadline'.",
      "info" : "A small penalty is inccured when a soft deadline is missed. A significant penalty is incurred when a hard deadline is missed.",
      "info" : "Simulation environment provides airports and flight routes modeled as a network graph (networkx library).",
      "info" : "Simulation environment provides a shortest path function based on the plane model: ObservationHelper.get_lowest_cost_path(state, airport1, airport2, plane_type: PlaneTypeID)",
      "info" : "ObservationHelper.available_destinations(state, airplane_obs, plane_type: PlaneTypeID)         # Returns available destination from an airport node.",
     ]
  },
  "additional-requirments": { [
       "requirement": "At each time step, the algorithm needs to assign actions to each aircraft. These actions are: unload cargo, load cargo, set destination (next airpor id), and set priority (aircraft).",
       "requirement": "Please base the assignment algorithm on a well known algorithm design pattern",
       "requirement": "In your design, please consider the results of the simulation data (see below) to help you make your decision.",
     ]
  }
"""

# llm_input = f"""
# {{
#   "task-objective": "Please design a cargo delivery routing algorithm for the Airlift Challenge 2.0 simulation environment. The primary objective is to meet the specified cargo delivery deadlines, and the secondary goal is to minimize the distance traveled.",
#   "simulation_environment":   {sim_description_verbose},
#   "simulation_data_analysis": {sim_data_analysis_summary},
#   "algorithm_requirements":   {alg_requirements_description_json}
# }}
# """

# summarized input
llm_input = f"""
{{
  **Task**
  Please design a cargo delivery schedueling and routing algorithm for the Airlift Challenge 2.0 simulation environment.
  
  **Objective**
  The primary objective is to meet the specified cargo delivery deadlines. The secondary objective is to minimize the distance traveled.
  
  **Algorithm Requirements**
  {alg_requirements_description_json}

  **Simulation Environment Description**
  {best_sim_env_summary}
  
  **Operational Insights - Simulation Data Analysis**
  {best_analysis_summary}
}}
"""


# llm_input = llm_input + "\n\nAction: Do not start writing yet, First explain everything I wanted you to do in this Prompt in Detail." 
# + "\n\nAction: Please write your algorithm design in Python. You may use any libraries you want, but please explain why you chose them and how they will be used to solve this problem.|end_of_text|.",

if PRINT_LLM_INPUT:
    print("\n\n---------------------------------------------------\nAlgorithm Design LLM Input:", llm_input)

response = ollama.chat(model=LLM_MODEL, messages=[
    {
      'role': 'system',
      'content': 'You are an expert software architect and a schedueling algorithm designer',     #1. set model context 
    },
    {
      'content': llm_input,
      'role': 'user',
    },
  ],
  options = {
    # 'logit_bias': LLM_LOGIT_BIAS,
    'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
  }
)


print("\n\n---------------------------------------------------\nLLM Output:\n")
alg_design = response['message']['content']
print("\n\nAlgorithm Design:\n",alg_design)

#
# "input_format": "",
# "output_format": "<Describe the desired format of output schedules>"



# """ Good summarization of the simulation environment:
#
# {
#   "Airport": {
#     "Limited Working Capacity": "Dictates number of aircraft that can be serviced simultaneously",
#     "Processing Order": [
#       "Unload cargo_items at destination",
#       "Load scheduled cargo_items",
#       "Fly to next airport if destination set"
#     ]
#   },
#   "Aircraft": {
#     "Actions": ["Unload", "Load", "Set Destination", "Set Priority"],
#     "States": ["WAITING", "PROCESSING", "MOVING", "READY_FOR_TAKEOFF"],
#     "Attributes": [
#       "Current Airport (ID)",
#       "Available Routes (IDs)",
#       "Cargo Onboard (List of IDs)",
#       "Cargo At Current Airport (List of IDs)",
#       "Current Weight",
#       "Max Weight",
#       "Max Range",
#       "Plane Type"
#     ],
#     "Next Action": {
#       "Tasks": ["Load Cargo", "Unload Cargo", "Set Destination", "Set Priority"],
#       "Priority": "Based on cargo items' hard deadlines"
#     },
#     "Processing Time": "Varies based on airport's working capacity"
#   },
#   "Route": {
#     "Representation": "Node/Edge graph (NetworkX)",
#     "Bidirectional": true,
#     "Distance": "Flight distance between two airports",
#     "Availability": {
#       "Disruptions": "Route becomes unavailable when 'mal' > 0"; Route available when 'mal' == 0,
#       "Options": ["Wait for route to be available", "Find alternative route"]
#     }
#   },
#   "Cargo Item": {
#     "Attributes": {
#       "Location": {"Current airport ID where the cargo_item is at": true},
#       "Destination": {"Airport ID to deliver cargo item": true},
#       "Weight": {"The weight of the cargo. Can only fit on aircraft if there is enough capacity": true},
#       "Earliest Pickup Time": {"The pickup time that cargo-item becomes available for pickup at the airport": true},
#       "Soft Deadline": {"Cargo-item should be delivered by this time": true},
#       "Hard Deadline": {"Significant penalty incurred if delivery is past this deadline": true, "No point delivering cargo items that pass this deadline": true},
#       "Is Available": {"Cargo item can be picked up when this condition is true": true}
#     }
# }
# """