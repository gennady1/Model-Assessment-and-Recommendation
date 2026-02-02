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
#from solution.mysolution import MySolution
from solution.mysolution_20240604_zephyr141b import Heuristic_Zephyr141b
from solution.mysolution_20240815_codestral22 import Solution_Codestral22



from solution.mysolution_20241129_hungarian import Solution_Hungarian
from airlift.solutions.baselines import ShortestPath, RandomAgent

# Helper methods
from airlift.solutions import doepisode
from eval_solution import write_results
from pprint import pprint as pp

#capturing result stats
import numpy as np
import pandas as pd


# Maximum number of steps the episode will run
max_cycles = 5000

# Use a plain map (this is faster to generate and captures essential elements of the scenario)

"""
Create an AirliftEnv using all the generators. There exist multiple generators for each aspect. For example instead of using the
DynamicCargoGenerator we can also use the StaticCargoGenerator.
"""

"""
Uncomment the scenario below that you would like to use.
"""


## A simple scenario with no dynamic events
plane_types = [
                PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)
                # , PlaneType(id=1, max_range=.6, speed=0.05, max_weight=4)
                # , PlaneType(id=2, max_range=.5, speed=0.03, max_weight=3)
            ]

def generate_env_one(num_of_aircraft_agents=8, working_capacity=2, aircraft_processing_time=60):
    env = AirliftEnv(
            world_generator=AirliftWorldGenerator(
              plane_types=plane_types,
              airport_generator=RandomAirportGenerator(
                  max_airports=10,
                  working_capacity=working_capacity,
                  processing_time=aircraft_processing_time,
                  make_drop_off_area=True,
                  make_pick_up_area=True,
                  num_drop_off_airports=3,
                  num_pick_up_airports=3,
                  mapgen=PlainMapGenerator()
    ,
              ),
              route_generator=RouteByDistanceGenerator(
                  route_ratio=2,
              ),
            cargo_generator=StaticCargoGenerator(
                  num_of_tasks=10,
                  max_weight=4,
                  soft_deadline_multiplier=10,
                  hard_deadline_multiplier=20,
              ),
              airplane_generator=AirplaneGenerator(num_of_agents=num_of_aircraft_agents),
              max_cycles=max_cycles
            ),
        )
    return env

def generate_env_dynamic_routes(num_of_aircraft_agents=2, working_capacity=2, airport_processing_time=0, route_ratio=2, poisson_lambda=.2, min_duration=10, max_duration=30):
    env = AirliftEnv(
            world_generator=AirliftWorldGenerator(
              plane_types=plane_types,
              airport_generator=RandomAirportGenerator(
                  max_airports=10,
                  working_capacity=working_capacity,
                  processing_time=airport_processing_time,
                  make_drop_off_area=True,
                  make_pick_up_area=True,
                  num_drop_off_airports=3,
                  num_pick_up_airports=3,
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
            cargo_generator=StaticCargoGenerator(
                  num_of_tasks=10,
                  max_weight=4,
                  soft_deadline_multiplier=10,
                  hard_deadline_multiplier=20,
              ),
              airplane_generator=AirplaneGenerator(num_of_agents=num_of_aircraft_agents),
              max_cycles=max_cycles
            ),
        )
    return env


SPEED_SIM_SLEEP_TIME = 0.001 # Set this to 0.1 to slow down the simulation

#run LLM gen code
# agent = RandomAgent()
# agent = ShortestPath()
agent = Heuristic_Zephyr141b()
# agent = Solution_Codestral22()
# agent = Solution_Hungarian()

env = generate_env_one(num_of_aircraft_agents=4)

env_info, metrics, time_taken, total_solution_time, step_metrics = doepisode(
    env,
    agent,
    render=False,
    render_sleep_time=SPEED_SIM_SLEEP_TIME,
    env_seed=100,
    solution_seed=200,
    capture_metrics=True)


print("Total Solution Time: ", total_solution_time)
print(metrics)
# print("\nStep Metrics: ", step_metrics)

#Metrics properties: 
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



# Questions:
#    1) How to get access to current time / frame from the environment?
#    2) Is it possible to get route flight time estimate (in terms of frame count?)
#    3) If yes (2), then does it include the processing time at each stop?
#    4) Alternative route...


# other Observation adding additional aircraft, pause