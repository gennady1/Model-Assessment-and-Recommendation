# Stripped my code from the 'Airlift challenge codebase'
# 20250312 - Adapted from: run_custom_scenario.py

import UniformCargoGenerator

# Environment
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs import PlaneType
from airlift.envs.generators.map_generators import PlainMapGenerator

# Generators
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.cargo_generators import StaticCargoGenerator

# Dynamic events
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.cargo_generators import DynamicCargoGenerator

# Starter kit solution
from solution.mysolution import MySolution
# from solution.mysolution_codestral22 import Solution_Codestral22
# from solution.mysolution_llm import Solution_LLM
from airlift.solutions.baselines import ShortestPath, RandomAgent

# Helper methods
from airlift.solutions import doepisode
from eval_solution import write_results

#capturing result stats
# import numpy as np
import pandas as pd
import os

#---------- Defaults -----------

# Set this to 0.1 to slow down the simulation
SPEED_SIM_SLEEP_TIME = 0.001

# Maximum number of steps the episode will run. 
# This value represents number of minutes in a day (60 * 24); we should consider modeling weekdays / weekends / holidays
max_cycles = 1440


##################
# RUN EXPERIMENT #
##################

RUN_EXPERIMENT_1 = True     # Test change in number of aircraft
RUN_EXPERIMENT_2 = True     # Test change in working capacities
RUN_EXPERIMENT_3 = True     # Test change in processing time
RUN_EXPERIMENT_4 = True     # Test change in poisson lambda value
RUN_EXPERIMENT_5 = True     # Test change in route unavailability min/max durration


##################
# DEFAULT VALUES #
##################


# There are 5 experiments evaluating 5 different attributes of the MOG. Each experiment (attributed being measured) is evaluated 20 times (num runs) 
RUNS = 50

#number of cargo items created for each run
NUM_TASKS = 113

#AIRPORTS
DROP_OFF_AIRPORTS = 27
PICK_UP_AIRPORTS = 1

num_of_aircraft_agents=30

working_capacity=10

airport_processing_time=135

#cargo generation parameters; offset from soft_deadline (uniformly distributed using the max_cycles)
earliest_pickup_offset = 200   # this value is used to subtract from soft_deadline to set earliest_pickup_time 
hard_deadline_offset = 300     # this value is used to add to soft_deadline to set hard_deadline 

#dynamic route factors
route_ratio = 2
poisson_lambda = .2
min_duration = 5
max_duration = 30


MAX_AIRPORTS = DROP_OFF_AIRPORTS + PICK_UP_AIRPORTS + 1     #MAX_AIRPORTS must be > DROP_OFF_AIRPORT + PICK_UP_AIRPORT

#create seperate folders for CSVs
csv_folder = "csv"
csv_path = os.path.join(os.getcwd(), csv_folder)
if not os.path.exists(csv_path):
    os.makedirs(csv_path)


# Use a plain map (this is faster to generate and captures essential elements of the scenario)

"""
Create an AirliftEnv using all the generators. There exist multiple generators for each aspect. For example instead of using the
DynamicCargoGenerator we can also use the StaticCargoGenerator.
"""

"""
Uncomment the scenario below that you would like to use.
"""
print(f"\n\n-----------------------------------------------------------\nGenerate scenario data (CSV format)\n  Configureation:" )
print(f"\tAirports: DROP_OFF_AIRPORTS = {DROP_OFF_AIRPORTS}, PICK_UP_AIRPORTS = {PICK_UP_AIRPORTS}, working_capacity = {working_capacity}, airport_processing_time = {airport_processing_time}")
print(f"\tAgents (aircraft) = {num_of_aircraft_agents}, NUM_TASKS = {NUM_TASKS}")
print(f"\tDynamic route configuration paramters: route_ratio = {route_ratio}, poisson_lambda = {poisson_lambda}, min_duration = {min_duration}, max_duration = {max_duration}")

## A simple scenario with no dynamic events
# for simplicicy sake, a one type of aircraft
plane_types = [
                PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)
                # , PlaneType(id=1, max_range=.6, speed=0.05, max_weight=4)
                # , PlaneType(id=2, max_range=.5, speed=0.03, max_weight=3)
            ]

def generate_env_one(num_of_aircraft_agents=5, working_capacity=2, airport_processing_time=60, num_of_tasks_cargoitems=50, route_ratio=2):
    env = AirliftEnv(
            world_generator=AirliftWorldGenerator(
              plane_types=plane_types,
              airport_generator=RandomAirportGenerator(
                  max_airports=MAX_AIRPORTS,
                  working_capacity=working_capacity,
                  processing_time=airport_processing_time,
                  make_drop_off_area=True,
                  make_pick_up_area=True,
                  num_drop_off_airports = DROP_OFF_AIRPORTS,
                  num_pick_up_airports  = PICK_UP_AIRPORTS,
                  mapgen=PlainMapGenerator()
    ,
              ),
              route_generator=RouteByDistanceGenerator(
                  route_ratio=route_ratio,
              ),
            cargo_generator=UniformCargoGenerator.UniformCargoGenerator(
                  num_of_tasks=num_of_tasks_cargoitems,
                  max_weight=4,
                  soft_deadline_multiplier=10,
                  hard_deadline_multiplier=20,
                  max_cycles=max_cycles,
                  earliest_pickup_offset = earliest_pickup_offset,       # this value is used to subtract from soft_deadline to set earliest_pickup_time 
                  hard_deadline_offset   = hard_deadline_offset,         # this value is used to add to soft_deadline to set hard_deadline 
              ),
              airplane_generator=AirplaneGenerator(num_of_agents=num_of_aircraft_agents),
              max_cycles=max_cycles
            ),
        )
    return env

def generate_env_dynamic_routes(num_of_aircraft_agents, working_capacity, airport_processing_time, route_ratio, poisson_lambda, min_duration, max_duration, num_of_tasks_cargoitems):
    env = AirliftEnv(
            world_generator=AirliftWorldGenerator(
              plane_types=plane_types,
              airport_generator=RandomAirportGenerator(
                  max_airports=MAX_AIRPORTS,
                  working_capacity=working_capacity,
                  processing_time=airport_processing_time,
                  make_drop_off_area=True,
                  make_pick_up_area=True,
                  num_drop_off_airports = DROP_OFF_AIRPORTS,
                  num_pick_up_airports  = PICK_UP_AIRPORTS,
                  mapgen=PlainMapGenerator()
    ,
              ),
            route_generator=RouteByDistanceGenerator(
                route_ratio=route_ratio,
                poisson_lambda=poisson_lambda,
                malfunction_generator=EventIntervalGenerator(
                    min_duration=min_duration,
                    max_duration=max_duration),
            ),
            cargo_generator=(UniformCargoGenerator)(
                  num_of_tasks=num_of_tasks_cargoitems,
                  max_weight=4,
                  soft_deadline_multiplier=10,
                  hard_deadline_multiplier=20,
                  max_cycles=max_cycles,
                  earliest_pickup_offset = earliest_pickup_offset,       # this value is used to subtract from soft_deadline to set earliest_pickup_time 
                  hard_deadline_offset = hard_deadline_offset,         # this value is used to add to soft_deadline to set hard_deadline 
              ),
              airplane_generator=AirplaneGenerator(num_of_agents=num_of_aircraft_agents),
              max_cycles=max_cycles
            ),
        )
    return env

agreggate_labels = [] # labels for agregate scores
aggregate_scores = [] # aggregate scores

labels = [] 
labels.append("Aircraft Quantity")
labels.append("wMOG capacity")
labels.append("Processing time")
labels.append("Steps to completion")
labels.append("Simulation time (sec)")
labels.append("Total flight distance")
labels.append("Total Lateness")
labels.append("Total waiting to process steps")
labels.append("Missed Deliveries")
labels.append("Score")



########## EXPERIMENT - 1 ##########
if RUN_EXPERIMENT_1:
    print(f"\n\n  Experiment-1: Simulating congestion by varying the number of aircraft (airport processing time={airport_processing_time}; wMOG={working_capacity})")
    results_data = []
    agg_res_data = []
    for i in range(RUNS):
        print(f"\tAircraft = {i+1}")

        #run LLM gen code
        # agent = RandomAgent()
        agent = ShortestPath()
        # agent = MySolution()
        # agent = Solution_Codestral22()
        # agent = Solution_LLM()

        env = generate_env_one(num_of_aircraft_agents=i+1, working_capacity=working_capacity, airport_processing_time=airport_processing_time, num_of_tasks_cargoitems=NUM_TASKS, route_ratio=route_ratio)

        env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
                        env,
                        agent,
                        render=False,
                        render_sleep_time=SPEED_SIM_SLEEP_TIME,
                        env_seed=100,
                        solution_seed=200,
                        capture_metrics=True)

        # Record results
        result = []

        result.append(i+1)                                      # aircraft_count
        result.append(working_capacity)                         # working capacity (wMOG)
        result.append(airport_processing_time)                  # airport processing time
        result.append(metrics.total_steps)                      # number of steps the simulation took to terminate  (all cargo items delivered, or the frame limit has been reached)
        result.append("{:.4f}".format(total_solution_time))     # sim time
        result.append(metrics.total_cost)                       # total flight distance traveled
        result.append(metrics.total_lateness)                   # total lateness (if the hard_deadline is passed -> it counts as a missed delivery)
        result.append(metrics.total_waiting_to_process_steps)   # total time steps it took for aircraft waiting to be processed.
        result.append(metrics.missed_deliveries)                # number of missed deliveries that missed hard deadline
        result.append(metrics.score)

        results_data.append(result)
        agg_res_data.append(metrics.score)

    # create labeled matrix (pandas dataframe); and save it as a CSV file
    df = pd.DataFrame(results_data, columns=labels)
    df.to_csv(csv_folder + "/results-1_aircraft-quantity.csv", index=False)

    # register scores only - used to generate score summary (result-0)
    agreggate_labels.append("Score  (Loss)\nAircraft Quantity")
    aggregate_scores.append(agg_res_data)
    

########## EXPERIMENT - 2 ##########
if RUN_EXPERIMENT_2:
    # Simulate increasing processing time
    print(f"Completed.\n  Experiment-2: Simulating congestion by varying the airport_capacity (wMOG); aircraft={num_of_aircraft_agents}; airport processing time={airport_processing_time}.")
    results_data = []
    agg_res_data = []
    for i in range(RUNS):
        print(f"\twMOG capacity = {i}")

        #run LLM gen code
        # agent = RandomAgent()
        agent = ShortestPath()
        # agent = MySolution()
        # agent = Solution_Codestral22()
        # agent = Solution_LLM()

        env = generate_env_one(num_of_aircraft_agents=num_of_aircraft_agents, working_capacity=i, airport_processing_time=airport_processing_time, num_of_tasks_cargoitems=NUM_TASKS, route_ratio=route_ratio)

        env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
                        env,
                        agent,
                        render=False,
                        render_sleep_time=SPEED_SIM_SLEEP_TIME,
                        env_seed=100,
                        solution_seed=200,
                        capture_metrics=True)

        # Record results
        result = []

        result.append(num_of_aircraft_agents)                   # aircraft_count
        result.append(i)                                        # working capacity (wMOG)
        result.append(airport_processing_time)                  # airport processing time
        result.append(metrics.total_steps)                      # number of steps the simulation took to terminate  (all cargo items delivered, or the frame limit has been reached)
        result.append("{:.4f}".format(total_solution_time))     # sim time
        result.append(metrics.total_cost)                       # total flight distance traveled
        result.append(metrics.total_lateness)                   # total lateness (if the hard_deadline is passed -> it counts as a missed delivery)
        result.append(metrics.total_waiting_to_process_steps)   # total time steps it took for aircraft waiting to be processed.
        result.append(metrics.missed_deliveries)                # number of missed deliveries that missed hard deadline
        result.append(metrics.score)   

        results_data.append(result)
        agg_res_data.append(metrics.score)

    # create labeled matrix (pandas dataframe); and save it as a CSV file
    df = pd.DataFrame(results_data, columns=labels)
    df.to_csv(csv_folder + "/results-2_wMOG.csv", index=False)

    # register scores only - used to generate score summary (result-0)
    agreggate_labels.append("Score  (Loss)\nWorking MOG Capacity")
    aggregate_scores.append(agg_res_data)

########## EXPERIMENT - 3 ##########  # Simulate increasing processing time
if RUN_EXPERIMENT_3:
    print(f"Completed.\n Experiment-3: Simulating congestion by varying the airport processing time (aircraft={num_of_aircraft_agents}; wMOG={working_capacity}).")
    results_data = []
    agg_res_data = []
    for i in range(RUNS):
        apt = i*10
        print(f"\tAirport processing capacity = {apt} steps")

        #run LLM gen code
        # agent = RandomAgent()
        agent = ShortestPath()
        # agent = MySolution()
        # agent = Solution_Codestral22()
        # agent = Solution_LLM()

        env = generate_env_one(num_of_aircraft_agents=num_of_aircraft_agents, working_capacity=working_capacity, airport_processing_time=apt, num_of_tasks_cargoitems=NUM_TASKS, route_ratio=route_ratio)

        env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
                        env,
                        agent,
                        render=False,
                        render_sleep_time=SPEED_SIM_SLEEP_TIME,
                        env_seed=100,
                        solution_seed=200,
                        capture_metrics=True)

        # Record results
        result = []

        result.append(num_of_aircraft_agents)                   # aircraft_count
        result.append(working_capacity)                         # working capacity (wMOG)
        result.append(apt)                                      # airport processing time
        result.append(metrics.total_steps)                      # number of steps the simulation took to terminate  (all cargo items delivered, or the frame limit has been reached)
        result.append("{:.4f}".format(total_solution_time))     # sim time
        result.append(metrics.total_cost)                       # total flight distance traveled
        result.append(metrics.total_lateness)                   # total lateness (if the hard_deadline is passed -> it counts as a missed delivery)
        result.append(metrics.total_waiting_to_process_steps)   # total time steps it took for aircraft waiting to be processed.
        result.append(metrics.missed_deliveries)                # number of missed deliveries that missed hard deadline
        result.append(metrics.score)   

        results_data.append(result)
        agg_res_data.append(metrics.score)

    # create labeled matrix (pandas dataframe); and save it as a CSV file
    df = pd.DataFrame(results_data, columns=labels)
    df.to_csv(csv_folder + "/results-3_processing-duration.csv", index=False)

    # register scores only - used to generate score summary (result-0)
    agreggate_labels.append("Score  (Loss)\nProcessing Time")
    aggregate_scores.append(agg_res_data)




#Route features: route_ratio=2, poisson_lambda=.2, min_duration=10, max_duration=30
labels = [] 
labels.append("Aircraft Quantity")
labels.append("wMOG capacity")
labels.append("Processing time")
labels.append("Route ratio")
labels.append("Poisson lambda")
labels.append("Min Duration")
labels.append("Max Duration")
labels.append("Steps to completion")
labels.append("Simulation time (sec)")
labels.append("Total flight distance")
labels.append("Total Lateness")
labels.append("Total waiting to process steps")
labels.append("Total waiting for route steps")
labels.append("Missed Deliveries")
labels.append("Score")

########## EXPERIMENT - 4 ##########
# Simulate increasing poisson lambda (route availability generation)
if RUN_EXPERIMENT_4:
    print(f"Completed.\n  Experiment-4: Simulating congestion by increasing poisson lambda value; aircraft={num_of_aircraft_agents}; wMOG={working_capacity}, airport processing time={airport_processing_time}")
    results_data = []
    agg_res_data = []
    for i in range(RUNS):

        #route factors
        # route_ratio = 2
        poisson_lambda = i * .02
        # min_duration = 5 + i*2
        # max_duration = min_duration + 1 + i*2                   #Needed to add 1; max duration has to be > than min_duration

        print(f"\tRoute ractors: route_ratio={route_ratio}, poisson_lambda={poisson_lambda}, min_duration={min_duration}, max_duration={max_duration}\n")

        #run LLM gen code
        # agent = RandomAgent()
        agent = ShortestPath()
        # agent = MySolution()
        # agent = Solution_Codestral22()
        # agent = Solution_LLM()

        
        env = generate_env_dynamic_routes(num_of_aircraft_agents, working_capacity, airport_processing_time, route_ratio, poisson_lambda, min_duration, max_duration, NUM_TASKS)
        # env = generate_env_two()

        env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
                        env,
                        agent,
                        render=False,
                        render_sleep_time=SPEED_SIM_SLEEP_TIME,
                        env_seed=100,
                        solution_seed=200,
                        capture_metrics=True)

        # Record results
        result = []
        result.append(num_of_aircraft_agents)                   # aircraft_count
        result.append(working_capacity)                         # wornking mog capacity at airfield
        result.append(airport_processing_time)                  # airport processing time
        result.append(route_ratio)                              # configuration parameter for the dynamic route
        result.append(poisson_lambda)                           # a chance distribution when route becomes unavailable
        result.append(min_duration)                             # min duration that route will be unavailable
        result.append(max_duration)                             # max duration route can be unavailable
        result.append(metrics.total_steps)                      # steps to completion of the sim run
        result.append("{:.4f}".format(total_solution_time))     # sim time
        result.append(metrics.total_cost)                       # total flight distance traveled
        result.append(metrics.total_lateness)                   # total lateness (if the hard_deadline is passed -> it counts as a missed delivery)
        result.append(metrics.total_waiting_to_process_steps)   # total time steps it took for aircraft waiting to be processed.
        result.append(metrics.total_waiting_for_route_steps)    # total time steps it took for aircraft waiting for route to be available.
        result.append(metrics.missed_deliveries)                # number of missed deliveries that missed hard deadline
        result.append(metrics.score)
        
        results_data.append(result)
        agg_res_data.append(metrics.score)

    # create labeled matrix (pandas dataframe); and save it as a CSV file
    df = pd.DataFrame(results_data, columns=labels)
    df.to_csv(csv_folder + "/results-4_dynamic-route-poisson.csv", index=False)

    # register scores only - used to generate score summary (result-0)
    agreggate_labels.append("Score (Loss)\nPoisson Lambda")
    aggregate_scores.append(agg_res_data)


########## EXPERIMENT - 5 ##########
# Simulate increasing processing time
if RUN_EXPERIMENT_5:
    print(f"Completed.\n  Experiment-5: Simulating congestion by route availabiltiy; aircraft={num_of_aircraft_agents}; wMOG={working_capacity}, airport processing time={airport_processing_time}")
    results_data = []
    agg_res_data = []
    for i in range(RUNS):

        #route factors
        # route_ratio = 2
        # poisson_lambda = i * .02
        min_duration = 5 + i*2
        max_duration = min_duration + 1 + i*2                   #Needed to add 1; max duration has to be > than min_duration

        print(f"\tRoute ractors: route_ratio={route_ratio}, poisson_lambda={poisson_lambda}, min_duration={min_duration}, max_duration={max_duration}\n")

        #run LLM gen code
        # agent = RandomAgent()
        agent = ShortestPath()
        # agent = MySolution()
        # agent = Solution_Codestral22()
        # agent = Solution_LLM()

        env = generate_env_dynamic_routes(num_of_aircraft_agents, working_capacity, airport_processing_time, route_ratio, poisson_lambda, min_duration, max_duration, NUM_TASKS)
        # env = generate_env_two()

        env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
                        env,
                        agent,
                        render=False,
                        render_sleep_time=SPEED_SIM_SLEEP_TIME,
                        env_seed=100,
                        solution_seed=200,
                        capture_metrics=True)

        # Record results
        result = []

        result.append(num_of_aircraft_agents)                   # aircraft_count
        result.append(working_capacity)                         # wornking mog capacity at airfield
        result.append(airport_processing_time)                  # airport processing time
        result.append(route_ratio)                              # configuration parameter for the dynamic route
        result.append(poisson_lambda)                           # a chance distribution when route becomes unavailable
        result.append(min_duration)                             # min duration that route will be unavailable
        result.append(max_duration)                             # max duration route can be unavailable
        result.append(metrics.total_steps)                      # steps to completion of the sim run
        result.append("{:.4f}".format(total_solution_time))     # sim time
        result.append(metrics.total_cost)                       # total flight distance traveled
        result.append(metrics.total_lateness)                   # total lateness (if the hard_deadline is passed -> it counts as a missed delivery)
        result.append(metrics.total_waiting_to_process_steps)   # total time steps it took for aircraft waiting to be processed.
        result.append(metrics.total_waiting_for_route_steps)    # total time steps it took for aircraft waiting for route to be available.
        result.append(metrics.missed_deliveries)                # number of missed deliveries that missed hard deadline
        result.append(metrics.score)

        results_data.append(result)
        agg_res_data.append(metrics.score)

    # create labeled matrix (pandas dataframe); and save it as a CSV file
    df = pd.DataFrame(results_data, columns=labels)
    df.to_csv(csv_folder + "/results-5_dynamic-route-minmax.csv", index=False)

    # register scores only - used to generate score summary (result-0)
    agreggate_labels.append("Score  (Loss)\nMax Duration")
    aggregate_scores.append(agg_res_data)

df_scores = pd.DataFrame([])
for idx, score_label in enumerate(agreggate_labels):
    df_scores[score_label] = aggregate_scores[idx]
df_scores.to_csv(f"{csv_folder}/results-0_{NUM_TASKS}ci-{num_of_aircraft_agents}a-sensitivity-analysis.csv", index=False)

print("Completed Experiments.")

#Additional metric properties: 
# Metrics
#   total_cost
#   total_scaled_cost
#   average_cost_per_plane
#   total_lateness
#   total_scaled_lateness
#   average_lateness_per_plane
#   total_steps
#   average_steps
#   total_waiting_steps
#   total_waiting_to_process_steps
#   total_waiting_for_route_steps
#   max_seconds_to_complete
#   total_malfunctions
#   missed_deliveries
#   total_rewards_for_all_agents
#   average_rewards_for_all_agents
#   score
#   total_cargo_generated
#   dynamic_cargo_generated