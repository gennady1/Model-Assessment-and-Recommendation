import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from collections import namedtuple

from pprint import pprint as pp
from functools import partial
from networkx import NetworkXNoPath


from airlift.solutions import Solution
from airlift.envs.agents import PlaneState
from airlift.envs.airport import Airport
from airlift.envs.plane_types import PlaneType
# from airlift.envs.route_map import RouteMap
from airlift.envs.airlift_env import ScenarioObservation as so 
from airlift.envs.airlift_env import ObservationHelper as oh, NOAIRPORT_ID

# from airlift.envs.airlift_env import ScenarioObservation as so     processing time:   so.processing_time
# from airlift.solutions.baselines import ShortestPath, RandomAgent

from solution.hungarian_alg import Hungarian_Matrix
# from sandbox.hungarian_alg import algorithm



LOG_TO_CONSOLE = True
# LOG_TO_CONSOLE = False

# get instanstance of Hungarian_Matrix
hm = Hungarian_Matrix()

class Solution_Hungarian(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    

    def __init__(self):
        super().__init__()

        self.cargo_items = None                     # a dictionary of active cargo_items;  c.id is the key
        self.cargo_delivered = None
        self.prioritized_cargo_items = None         #a sorted list of self.cargo_items; the sort order is hardest_deadline, then earliest pickup time
        self.cargo_delivery_path = None             #dictionary: cargo_id : {'path', 'cost'}

        self.cost_routes_available = None           # plane type specific; flyable routes - can be flown now
        self.cost_routes_not_available = None       # plane type specific; flyable route currently not available
        
        self.plane_types = None                     # type of aircraft objects in the scenario
        self.airplane_id_list = None                # aircraft ids
        self.aircraft = None                        # dictionary of aircraft agent objects
        
        self.cargo_item_to_aircraft_assignment = None                  # a dictionary of tasks for each cargo item, aircraft and actions

        self.global_flight_path_graph = None
        
        self.full_delivery_paths = None
        self.plane_graph = None
        self.view = None
        self.global_view = None

        self.processing_time = 0                    # the time it takes to process an aircraft after it lands, prior to being tasked

        self.aircraft_state_lookup = {
            0: 'WAITING',                           # Airfiled at capcity, busy processing other aircraft.
            1: 'PROCESSING',                        # Aircraft is being processed. The processing duration gets read from the environment in the reset() step
            2: 'MOVING',                            # Aircraft is flying en route
            3: 'READY_FOR_TAKEOFF'                  # Aircraft is ready for takeoff
        }

        # self.STOP_IN_STEPS = 200
        # self.CURRENT_STEP = 0

        if LOG_TO_CONSOLE:
            print("\n\n-------mysolution.init()-------")
        


    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        
        super().reset(obs, observation_spaces, action_spaces, seed)

        #manually keeping track of time steps
        self._elapsed_steps = 0

        # a list of cargo items, sorted by earliest hard deadline, and then by earliest pickup time.
        self.prioritized_cargo_items = []

        # a dictionary for keeping sorted list of cargo items and their properties and aircraft assignment. 
        self.cargo_delivery_path = {}

        # a dictionary of tasks for each cargo item, aircraft and actions
        self.cargo_item_to_aircraft_assignment = {}

        # cargo delivery paths (source -> destination) using shortest path alg.
        self.full_delivery_paths = {}
        
        self.cargo_delivered = []

        self.plane_types = []

        # a dictionary of valid flight routes for each aircraft type (edge graph)
        self.view = {}

        #note: whats the difference between self.view and plane_graph
        self.plane_graph = {}

        self.cost_routes_available = {}
        self.cost_routes_not_available = {}

        # get state
        state = self.get_state(obs)

        # the time it takes to process an aircraft after it lands
        self.processing_time = state['scenario_info'][0].processing_time

        # Retrieve the list of airplanes IDs from the observation, which is a dictionary with airplane IDs as keys (provided by the EnvAgent)
        self.airplane_id_list = list(obs.keys())     #EnvAgent; agent keys

        self.aircraft = {}
        for a in self.airplane_id_list:

            #build a list of aircraft from observation object
            self.aircraft[a] = obs[a]

            # Build a list of plane types (integers); plane_type is used to calculate a graph for each plane type
            if self.aircraft[a]['plane_type'] not in self.plane_types:
                self.plane_types.append(self.aircraft[a]['plane_type'])

    
        # build a flight graph for each plane type; not sure if this changes when mal>0
        #should compare against self.view[airplane_type]
        for i in state['route_map']:
            self.plane_graph[i] = state['route_map'][i]

        #
        self.build_global_flight_path_graph(state)

        if LOG_TO_CONSOLE:
            print("\n\n-------mysolution.reset()-------")
            

        return



    def policies(self, obs, dones, infos):

        #get the global state  (pulls the data from the first agent / flight)
        state = self.get_state(obs)

        # update route lookup tables
        self.build_route_lookup_tables()

        # Build a flight route map for each airplane type (short range aircraft cant fly long range routes)
        # routes are effected by mal propery; needs to be recomputed every time step
        self.build_route_path_for_each_plane_type()
        # for plane_type in self.plane_types:
        #     self.print_digraph_to_console(self.view[plane_type])
        #     self.plot_digraph(plane_type)

        # update active cargo items
        self.cargo_items = {}
        for ci in state["active_cargo"]:
            self.cargo_items[ci.id]=ci

        # Workflow pipeline for the task assignment problem is:
        #   1) prioritize available cargo items,
        #   2) construct the cost matrix for aircraft and cargo items,
        #   3) assign cargo items to aircraft


        # Workflow pipeline 1: sort cargo_items based on their hard_delivery time, then by earliest pickup time
        self.prioritized_cargo_items = sorted(self.cargo_items.values(), key=lambda ci: (ci.hard_deadline, ci.earliest_pickup_time))

        # Workflow pipeline 2: construct the cost matrix for aircraft and cargo items,
        cost_matrix = self.create_cost_matrix_and_solve_it()

        # Workflow pipeline 3: solve the assignment
        
        assert False

        # task aircraft to move cargo
        actions = {}

        # update the time step
        self._elapsed_steps+=1
        # print(self._elapsed_steps)

        return actions

    
    # def create_cost_matrix_and_solve_it(self):

    #     # # create a list of cargo items that are in current window#
    #     # valid_cargo_items = []
    #     # for ci in self.prioritized_cargo_items:
    #     #     if ci.hard_deadline > self._elapsed_steps and self._elapsed_steps > ci.earliest_pickup_time - 100:
    #     #         valid_cargo_items.append(ci)

    #     # Initialize cost dictionary
    #     G = {}

    #     # Determine number of rows and columns in the square matrix
    #     num_rows = max(len(self.aircraft), len(self.prioritized_cargo_items))
        
    #     # Add fake aircraft if necessary to make the matrix square
    #     fake_aircraft_ids = [f'f_{i}' for i in range(num_rows - len(self.aircraft))]
    #     all_aircraft_ids = list(self.aircraft.keys()) + fake_aircraft_ids

    #     # Iterate over each aircraft (worker) and cargo item (job)
    #     for i, aircraft_id in enumerate(all_aircraft_ids):
    #         if aircraft_id.startswith('f_'):  # Fake aircraft
    #             G[aircraft_id] = {f'c{j}': 0.0001 for j in range(num_rows)}
    #         else:
    #             aircraft = self.aircraft[aircraft_id]
    #             if aircraft['state'] == PlaneState.READY_FOR_TAKEOFF:
    #                 G[aircraft_id] = {}  # Initialize the dictionary for this aircraft
    #                 for j, ci in enumerate(self.prioritized_cargo_items):
    #                     # Calculate the cost of assigning the aircraft to the cargo item
    #                     pickup_cost_dict = self.get_path_cost(aircraft['current_airport'], ci.location, aircraft['plane_type'])
    #                     delivery_cost_dict = self.get_path_cost(ci.location, ci.destination, aircraft['plane_type'])

    #                     if pickup_cost_dict is not None and delivery_cost_dict is not None:
    #                         # Calculate the total cost
    #                         total_cost = pickup_cost_dict['cost'] + delivery_cost_dict['cost']

    #                         # Adjust cost based on cargo priority (lower index means higher priority)
    #                         total_cost *= (j / len(self.prioritized_cargo_items))  # Lower priority items have a higher multiplier

    #                         # Set minimum cost to avoid division by zero
    #                         if total_cost == 0:
    #                             total_cost = 0.0001

    #                         # Update the cost dictionary
    #                         G[aircraft_id][f'c{j}'] = total_cost

    #                 # Add fake cargo items if necessary
    #                 for j in range(len(self.prioritized_cargo_items), num_rows):
    #                     G[aircraft_id][f'c{j}'] = 0
        
    #     #we now have cost matrix
    #     if LOG_TO_CONSOLE:
    #         print("CostMatrix:")
    #         pp(G)

    #     res = algorithm.find_matching(G, matching_type = 'min', return_type = 'list')
        
       

    #     if LOG_TO_CONSOLE:
    #         print("Solution:")
    #         pp(res)

    #     return res


    def create_cost_matrix_and_solve_it(self):
        ''' This method generated a 2D cost matrix then used the hungarian solver implemented by:
        https://python.plainenglish.io/hungarian-algorithm-introduction-python-implementation-93e7c0890e15
        '''

        # # create a list of cargo items that are in current window#
        # valid_cargo_items = []
        # for ci in self.prioritized_cargo_items:
        #     if ci.hard_deadline > self._elapsed_steps and self._elapsed_steps > ci.earliest_pickup_time - 100:
        #         valid_cargo_items.append(ci)

        # Initialize cost matrix with infinity values
        num_aircraft = len(self.aircraft)
        num_cargo_items = len(self.prioritized_cargo_items)
        max_dim = max(num_aircraft, num_cargo_items)
        cost_matrix = [[float('inf')] * max_dim for _ in range(max_dim)]

        # Iterate over each aircraft (worker) and cargo item (job)
        for i, aircraft_id in enumerate(self.aircraft):
            aircraft = self.aircraft[aircraft_id]
            if aircraft['state'] == PlaneState.READY_FOR_TAKEOFF:
                for j, ci in enumerate(self.prioritized_cargo_items):
                    # Calculate the cost of assigning the aircraft to the cargo item
                    pickup_cost_dict = self.get_path_cost(aircraft['current_airport'], ci.location, aircraft['plane_type'])
                    delivery_cost_dict = self.get_path_cost(ci.location, ci.destination, aircraft['plane_type'])

                    if pickup_cost_dict is not None and delivery_cost_dict is not None:
                        # Calculate the total cost
                        total_cost = pickup_cost_dict['cost'] + delivery_cost_dict['cost']

                        # Adjust cost based on cargo priority (lower index means higher priority)
                        total_cost *= (j / len(self.prioritized_cargo_items))  # Lower priority items have a higher multiplier

                        # Update the cost matrix
                        cost_matrix[i][j] = total_cost


        #we now have a square (padded) cost matrix, solve the cost_matrix using the hungarian algorithm
        np_cost_matrix = np.array(cost_matrix)
        
        #for logging
        row_labels = ""
        col_labels = ""

        if LOG_TO_CONSOLE:
            r_size = len(cost_matrix)
            if r_size == 0:
                c_size=0
            else:
                c_size = len(cost_matrix[0])

            #lets quickly build a labeled matrix
            row_labels = [i for i in self.aircraft]
            while len(row_labels) < max_dim:
                row_labels.append('None')

            col_labels = [j.id for j in self.prioritized_cargo_items]
            while len(col_labels) < max_dim:
                col_labels.append('None')

            labeled_cm = pd.DataFrame(cost_matrix, index=row_labels, columns=col_labels)
            print(f"Cost matrix: {num_aircraft} x {num_cargo_items} (#aircraft x #cargo_items); Final dim: {r_size}x{c_size}:\n{labeled_cm}")
            

        #solve the aircraft to cargo assignment using hungarian algorithm
        ans_pos = hm.hungarian_algorithm(np_cost_matrix.copy()) #Get the element position.
        ans, ans_mat = hm.ans_calculation(np_cost_matrix, ans_pos) #Get the minimum or maximum value and corresponding matrix.


        if LOG_TO_CONSOLE:
            labeled_cm = pd.DataFrame(cost_matrix, index=row_labels, columns=col_labels)
            print(f"Cost matrix: {num_aircraft} x {num_cargo_items} (#aircraft x #cargo_items); Final dim: {r_size}x{c_size}:\n{labeled_cm}")
            labeled_am = pd.DataFrame(ans_mat, index=row_labels, columns=col_labels)
            print(f"Cost Matrix size: {num_aircraft}(aircraft) x {num_cargo_items}(cargo_items); solution cost = {ans} ; Adjusted size: {r_size}x{c_size}:\n{labeled_am}")

        return ans_mat
    
                    

    def print_state_info(self):
        '''
        Just a simple output to console: aircraft, cargo items, 
        '''
        print("print_state_info()\n  AIRCRAFT")
        for k, v in self.aircraft.items():
            print(f"\tid={k}: [state={self.aircraft_state_lookup[v['state']]}, current_airport={v['current_airport']}, current_weight={v['current_weight']}, destination={v['destination']}, cargo_onboard={v['cargo_onboard']}, available_routes={v['available_routes']}")

        print(f"  CARGO\tID sorted order: {[pci.id for pci in self.prioritized_cargo_items]}")
        
        for c in self.cargo_items.values():
            paths = ""
            for pt, d in self.cargo_delivery_path[c.id].items():
                paths += f"  \t{self.airplane_id_list[pt]}:{c.location}"
                for p in d['path']:
                    paths+=f"→{p}"
                paths+=f", cost={round(d['cost'], 3)}"
                
            print(f"\tid={c.id}, available={c.is_available}, location={c.location}, destination={c.destination}, hard_deadline={c.hard_deadline}, earliest_pickup_time={c.earliest_pickup_time}, delivery paths: {paths}")
    
    def build_route_lookup_tables(self):
        '''
        Simple helper funciton that builds the look up tables for routes available (mal==0) and routes not available (mal>0)
        '''

        # build a list of available (and not available) routes for each plane_type
        for plane_type in self.plane_types:

            #initialize dictionary
            mal_good = {}
            mal_bad = {}
            
            # check each edge where 'mal' > 0
            for u, v, attr in self.plane_graph[plane_type].edges(data=True):
                if attr['mal']==0:
                    mal_good[(u, v)] = attr['cost']
                else:
                    mal_bad[(u, v)] = attr['cost']

            self.cost_routes_available[plane_type] = mal_good
            self.cost_routes_not_available[plane_type] = mal_bad

        # print("\nValid Routes")
        # pp(self.valid_routes)

        # print("\nInvalidRoutes")
        # pp(self.not_valid_routes)

    

    # ideally this step should only be called when there is change in route, or change in aircraft schedules..
    def calculate_cargo_delivery_paths(self):
        '''
        Calculates a shortest path to cargo destination
        cargo.location can become 0 (unavailable) when loaded on aircraft. Solution is to only update the paths if cargo.location is not 0
        '''

        #ensure that (all) cargo_item are not at their final destination
        assert all(c.location != c.destination for c in self.cargo_items.values())
        
        # calculate delivery path for each cargo item (shortest distance); class is airlift.envs.airlift_env.CargoObservation
        for c in self.cargo_items.values():

            #calculate plane specific route to destination
            path_cost = {}

            if c.location != 0:     #cargo.location==0 -> cargo is on aircraft; this will be recalculated when aircraft lands /before any actions are generated.
                for pt in self.plane_types:
                    path_cost[pt] = self.get_path_cost(c.location, c.destination, pt)

                self.cargo_delivery_path[c.id] = path_cost

        # print("\nDebug: Cargo Delivery Paths")
        # pp(self.cargo_delivery_path)
        # assert False
        
        return
    
    def get_path_cost(self, start: Airport, end: Airport, plane: PlaneType):
        '''
        Returns a summed cost of edges from start airport to end airport (using shortest path algorithm).
        There should be similar function in route_map.get_flight_cost(); available in the env class, but not 

        This function is now recursive, in the case that there is no path from start to end, the algorithm will try to move closer to the end as it can (by walking backwards recursively from destination)
        '''
        
        if start == end:
            return {'path':[], 'cost': 0}

        elif start==NOAIRPORT_ID or end==NOAIRPORT_ID:
            return {'path':[], 'cost': float('inf')}

        path = []
        
        #get shortest path
        try:
            path = nx.shortest_path(self.view[plane], start, end, weight="cost")[1:]
        except nx.NetworkXNoPath as e:
            # print(e)
            path = nx.shortest_path(self.global_view, start, end, weight="cost")[1:]
            if len(path)>1:
                res = self.get_path_cost(start, path[-2], plane)
            else:
                res = {'path':[], 'cost': float('inf')}

            # print(f"\n{e} Global_path={start}→{path}; best subpath={start}→{res['path']} cost={round(res['cost'], 3)}")
            return res 

        #calculate cost of edges along the path
        cost = 0
        start_location = start
        for p in path:
            cost += self.cost_routes_available[plane][(start_location, p)]
            start_location = p

        return {'path':path, 'cost':cost}

    def build_route_path_for_each_plane_type(self):
        '''
        Builds route path (subgraph) view for each plane type. Route available is when mal==0. This function needs to be recomputed every time step.
        Often smaller aircraft have shorter range; which means that they may not be able to fly long segments between airports)
        self.plane_graph[plane_type]: The key 'plane_type' is an integer that designates type of aircraft. The value is networkx.classes.digraph.DiGraph object.
        The self.filter_edge is a call to a function that will filter the edges where the 'mal' property set to 0 (the time remainign time for route malfunction, 0 -> route available).
        '''
        
        # Add the subgraph view for each plane_type
        for plane_type in self.plane_types:

            # Used for passing additional argument `plane_type` to the inner method
            partial_function = partial(self.filter_edge, plane_type)

            # create a view where flight route is available
            self.view[plane_type] = nx.subgraph_view(self.plane_graph[plane_type], filter_edge=partial_function)
            

    def print_digraph_to_console(self, digraph_obj):
        ''' Simple method for printing out graph object. '''
        print("\n\n", digraph_obj, "\tEdges in the graph:")
        for u, v, attr in digraph_obj.edges(data=True):
            print(f"Edge from {u} to {v}; attributes: {attr}")

    def plot_digraph(self, plane_type):
        '''Plots a flight graph for specific plane_type'''
        
        digraph_obj = self.view[plane_type]
        pos = nx.spring_layout(digraph_obj)
        nx.draw(digraph_obj, pos, with_labels=True)

        #for display purpose only, round the edge costs
        rounded_labels = {}
        for k, v in self.cost_routes_available[plane_type].items():
            rounded_labels[k] = round(v, 3)
        
        nx.draw_networkx_edge_labels(digraph_obj, pos, edge_labels=rounded_labels)
        plt.show()


    def filter_edge(self, plane_type, u, v):
        """  These function filter edges in a graph based on certain conditions (e.g., if an edge is currently unavailable due to maintenance). """
        return self.plane_graph[plane_type][u][v]['mal'] == 0

    def build_global_flight_path_graph(self, state):
        ''' Used to calculate shortest path for cargo items '''
        self.global_flight_path_graph = oh.get_multidigraph(state)
        self.global_view = nx.subgraph_view(self.global_flight_path_graph, filter_edge=self.filter_multi_graph_edge)
        # self.print_digraph_to_console(self.global_view)
    
    def filter_multi_graph_edge(self, u, v, key):
        """ Filter the MultiDiGraph (created from collection of DiGraphs) """
        return self.global_flight_path_graph[u][v][key]['mal'] == 0




######################### OLD CODE #########################
# def assign_cargo_item_to_aircraft(self):

#         # create a list of cargo items that are in current window#
#         valid_cargo_items = []
#         for ci in self.prioritized_cargo_items:
#             if ci.hard_deadline > self._elapsed_steps and self._elapsed_steps > ci.earliest_pickup_time - 100:
#                 valid_cargo_items.append(ci)

#         # Initialize cost matrix with infinity values
#         num_aircraft = len(self.aircraft)
#         num_cargo_items = len(valid_cargo_items)
#         max_dim = max(num_aircraft, num_cargo_items)
#         cost_matrix = [[float('inf')] * max_dim for _ in range(max_dim)]
        

#         # Iterate over each aircraft (worker) and cargo item (job)
#         for i, aircraft_id in enumerate(self.aircraft):
#             aircraft = self.aircraft[aircraft_id]
#             if aircraft['state'] == PlaneState.READY_FOR_TAKEOFF:
#                 for j, ci in enumerate(valid_cargo_items):

#                     # Added correction
#                     pickup_path_cost = self.get_path_cost(aircraft['current_airport'], ci.location, aircraft['plane_type'])
#                     delivery_path_cost = self.get_path_cost(ci.location, ci.destination, aircraft['plane_type'])

#                     # Check if the aircraft can reach the cargo item's location (path) && can take at least one step; 
#                     if ci.location in pickup_path_cost['path'] and len(delivery_path_cost['path'])>0:
#                         # Calculate the cost of assigning the aircraft to the cargo item
#                         pickup_cost = pickup_path_cost['cost']
#                         delivery_cost = delivery_path_cost['cost']
#                         total_cost = pickup_cost + delivery_cost

#                         # Adjust cost based on cargo priority and route availability
#                         if ci.hard_deadline < ci.soft_deadline:
#                             total_cost *= 2  # Double the cost for hard deadline cargo items
#                         if len(aircraft['available_routes']) < max_dim:
#                             total_cost += (max_dim - len(aircraft['available_routes'])) * 10  # Add penalty for limited route availability

#                         # Update the cost matrix
#                         cost_matrix[i][j] = total_cost

#         if LOG_TO_CONSOLE:
#             r_size = len(cost_matrix)
#             if r_size == 0:
#                 c_size=0
#             else:
#                 c_size = len(cost_matrix[0])
            
#             print(f"Cost matrix: {num_aircraft} x {num_cargo_items} (#aircraft x #cargo_items); Final dim: {r_size}x{c_size} ")

#         return cost_matrix