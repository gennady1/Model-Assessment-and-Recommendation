#mysolution_codestral22

import networkx as nx
import matplotlib.pyplot as plt


from pprint import pprint as pp
from functools import partial
from networkx import NetworkXNoPath


from airlift.solutions import Solution
# from airlift.envs.agents import PlaneState
from airlift.envs.airport import Airport
from airlift.envs.plane_types import PlaneType
# from airlift.envs.route_map import RouteMap
from airlift.envs.airlift_env import ObservationHelper as oh, NOAIRPORT_ID

# from airlift.envs.airlift_env import ScenarioObservation as so     processing time:   so.processing_time

# from airlift.solutions.baselines import ShortestPath, RandomAgent

LOG_TO_CONSOLE = True
# LOG_TO_CONSOLE = False



class Solution_Codestral22(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    

    def __init__(self):
        super().__init__()

        self.cargo_items = None                     # a dictionary of active cargo_items;  c.id is the key
        self.cargo_delivered = None
        self.prioratized_cargo_items = None         #a sorted list of self.cargo_items; the sort order is hardest_deadline, then earliest pickup time
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

        self.aircraft_state_lookup = {
            0: 'WAITING',
            1: 'PROCESSING',
            2: 'MOVING',
            3: 'READY_FOR_TAKEOFF'
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
        self.prioratized_cargo_items = []

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

        #sort cargo_items based on their hard_delivery time, then by earliest pickup time
        self.prioratized_cargo_items = sorted(self.cargo_items.values(), key=lambda ci: (ci.hard_deadline, ci.earliest_pickup_time))

        # task aircraft to move cargo
        actions = self.assign_actions_for_cargo_v1(state)
        # actions = self.assign_actions_for_cargo_v2(state)

        # update the time step
        self._elapsed_steps+=1
        # print(self._elapsed_steps)

        return actions

    # this is the second version of the generated code
    def assign_actions_for_cargo_v2(self, state):
        # Sort cargo items based on their earliest pickup time
        self.prioratized_cargo_items = sorted(self.cargo_items.values(), key=lambda ci: (ci.earliest_pickup_time))

        actions = {}

        for aircraft_id, aircraft in self.aircraft.items():
            cargo_to_unload = []
            cargo_to_load = []

            # If the aircraft is at an airport, check for available cargo items to load
            if aircraft['current_airport'] > 0:
                for ci in self.prioratized_cargo_items:
                    cost_path = self.get_path_cost(aircraft['current_airport'], ci.destination, aircraft['plane_type'])
                    if ci.location == aircraft['current_airport'] and ci.is_available and aircraft['current_weight'] + ci.weight <= aircraft['max_weight']:
                        cargo_to_load.append(ci.id)
                        # Update the current weight of the aircraft
                        aircraft['current_weight'] += ci.weight

                        # DEBUG ADD: 
                        if LOG_TO_CONSOLE:
                            print(f"\nACTION (t={self._elapsed_steps}): {aircraft_id}-LOAD_CARGO & SET_DESTINATION; Cargo_Item-{ci.id}; {aircraft['current_airport']} → {cost_path['path'][0]}")

                        break
                self.prioratized_cargo_items = [ci for ci in self.prioratized_cargo_items if ci.id not in cargo_to_load]

            # If the aircraft is en route, check for cargo items to unload
            else:
                for ci_id in aircraft['cargo_onboard']:
                    ci = self.cargo_items[ci_id]
                    cost_path = self.get_path_cost(aircraft['current_airport'], ci.destination, aircraft['plane_type'])
                    if cost_path['path']==[] or aircraft['current_airport'] == cost_path['path'][-1]:
                        cargo_to_unload.append(ci_id)
                        # Update the current weight of the aircraft
                        aircraft['current_weight'] -= ci.weight

                         # DEBUG ADD:
                        if LOG_TO_CONSOLE:
                            print(f"\nACTION (t={self._elapsed_steps}): UNLOAD; {aircraft_id}-UNLOAD; Cargo_Item-{ci_id}; Airport-{aircraft['current_airport']}; soft_dealine={ci.soft_deadline}, hard_deadline={ci.hard_deadline}")

                aircraft['cargo_onboard'] = [ci for ci in aircraft['cargo_onboard'] if ci not in cargo_to_unload]

            # If there are no cargo items to unload or load, set the destination as the current airport
            if len(cargo_to_unload) == 0 and len(cargo_to_load) == 0:
                destination = aircraft['current_airport']
            else:
                # If there are cargo items to unload or load, set the destination as their common destination
                destinations = [self.cargo_items[ci].destination for ci in cargo_to_unload + cargo_to_load]
                destination = max(set(destinations), key=destinations.count) if destinations else aircraft['current_airport']

            actions[aircraft_id] = {
                'cargo_to_unload': cargo_to_unload,
                'cargo_to_load': cargo_to_load,
                'destination': destination,
                'priority': 1}

        return actions


    # this is the first version of the generated code
    # This implementation assumes that `self.find_next_closest_airport(source, target)` and `self.check_route_availability(path)` are already implemented functions.
    # The function `self.assign_aircraft_to_cargo(self, state)` will return a dictionary of action items for each aircraft based on the current state of the cargo items, aircraft, and available routes.
    def assign_actions_for_cargo_v1(self, state):
        actions = {}

        # Iterate through each aircraft
        for aircraft_id in self.aircraft:
            aircraft = self.aircraft[aircraft_id]
            cargo_to_unload = []
            cargo_to_load = []
            destination = NOAIRPORT_ID                                                      #REPAIR: replaced None with NOAIRPORT_ID
            priority = 0
            

            if aircraft['current_airport'] > 0:                                             #REPAIR added check condition that ensures the aircraft is at the airport (i.e., not moving)

                # Check if the aircraft is at its current airport and unload any cargo that needs to be delivered there
                for cargo_id in aircraft['cargo_onboard']:               

                    try:
                        ci = self.cargo_items[cargo_id]
                        cost_path = self.get_path_cost(ci.location, ci.destination, aircraft['plane_type'])                         #REPAIR #registred plane specific delivery path to destination
                        self.cargo_delivery_path[cargo_id] = cost_path            
                    
                        if cost_path['path']==[] or aircraft['current_airport'] == cost_path['path'][-1]:                           #REPAIR: Added quotes
                            cargo_to_unload.append(ci.id)
                            aircraft['current_weight'] -= ci.weight        #INFO: this is probably done by the environment

                            if LOG_TO_CONSOLE:
                                print(f"\nACTION (t={self._elapsed_steps}): UNLOAD; {aircraft_id}-UNLOAD; Cargo_Item-{cargo_id}; Airport-{aircraft['current_airport']}; soft_dealine={ci.soft_deadline}, hard_deadline={ci.hard_deadline}")
                    except KeyError:
                        # pass
                        print(f"\n\tLookupFailure-cargo_item: KeyNotFound: cargo_key:{cargo_id}:{self.cargo_items.keys()}")


                # Iterate through each cargo item in priority order
                for idx, pci in enumerate(self.prioratized_cargo_items):
                    # ci = self.cargo_items[cargo_item]                                                                         #REPAIR replaced index with the object from for loop declartion

                    #check to load objects at current airport
                    if aircraft['current_airport'] == pci.location and pci.is_available and pci.location != pci.destination and aircraft['current_weight'] + pci.weight <= aircraft['max_weight']:
                        cost_path = self.get_path_cost(pci.location, pci.destination, aircraft['plane_type'])
                        if len(cost_path['path'])>0:
                            cargo_to_load.append(pci.id)
                            destination = cost_path['path'][0]
                            aircraft['current_weight'] += pci.weight
                            if LOG_TO_CONSOLE:
                                    print(f"\nACTION (t={self._elapsed_steps}): {aircraft_id}-LOAD_CARGO & SET_DESTINATION; Cargo_Item-{pci.id}; {aircraft['current_airport']} → {cost_path['path'][0]}")

                    # move aircraft to airport with cargo item
                    cost_path = self.get_path_cost(aircraft['current_airport'], pci.location, aircraft['plane_type'])
                    

                    # Check if the aircraft can carry more weight
                    if pci.is_available and aircraft['current_weight'] + pci.weight <= aircraft['max_weight']:      #REPAIR added 'is_available' check

                        # this is aircraft can move the cargo item closer to destination.
                        if len(cost_path['path'])>0:

                            #only task aircraft that is closest to the cargo item
                            if pci.id not in self.cargo_delivery_path.keys() or cost_path['cost'] < self.cargo_delivery_path[pci.id]['cost']:
                                self.cargo_delivery_path[pci.id] = cost_path

                            destination = self.cargo_delivery_path[pci.id]['path'][0]
                            priority = len(self.prioratized_cargo_items) - idx

                            if LOG_TO_CONSOLE:
                                print(f"\nACTION (t={self._elapsed_steps}): {aircraft_id}-MOVE: {self.aircraft[aircraft_id]['current_airport']} → {self.cargo_delivery_path[pci.id]['path']}; Pick Cargo_Item-{pci.id} @ airport {pci.location}")

                        

                        # Check if the route is available
                        # if self.check_route_availability(path):
                        #     cargo_to_load.append(ci.id)
                        #     aircraft['current_weight'] += ci.weight
                        #     destination = path[-1]
                        #     priority = max(priority, ci.hard_deadline - state.time)

                actions[aircraft_id] = {
                    "cargo_to_unload": cargo_to_unload,
                    "cargo_to_load": cargo_to_load,
                    "destination": destination,
                    "priority": priority
                }
        return actions

    
                    


    def print_state_info(self):
        '''
        Just a simple output to console: aircraft, cargo items, 
        '''
        print("print_state_info()\n  AIRCRAFT")
        for k, v in self.aircraft.items():
            print(f"\tid={k}: [state={self.aircraft_state_lookup[v['state']]}, current_airport={v['current_airport']}, current_weight={v['current_weight']}, destination={v['destination']}, cargo_onboard={v['cargo_onboard']}, available_routes={v['available_routes']}")

        print(f"  CARGO\tID sorted order: {[pci.id for pci in self.prioratized_cargo_items]}")
        
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