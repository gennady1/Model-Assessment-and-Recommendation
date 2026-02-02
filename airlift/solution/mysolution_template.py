# Questions:
#    1) How to get access to current time / frame from the environment?
#    2) Is it possible to get route flight time estimate (in terms of frame count?)
#    3) If yes (2), then does it include the processing time at each stop?
#    4) Alternative route...


# other Observation adding additional aircraft, pause

from math import ceil

import networkx as nx
import matplotlib.pyplot as plt

# this library is used for route cache implementation,
# This is naive implementation as it recalculates route paths once per every simulation step.
# More efficient implementation would be create a look up, only update route paths when route availability changes.
import functools


from pprint import pprint as pp
from functools import partial
from networkx import NetworkXNoPath

from airlift.solutions import Solution
from airlift.envs.airport import Airport
from airlift.envs.plane_types import PlaneType
from airlift.envs.airlift_env import ObservationHelper as oh, NOAIRPORT_ID 

# from airlift.envs import ActionHelper
# from airlift.envs.agents import PlaneState
# from airlift.envs.route_map import RouteMap

# Should consider using ObservationHelper and ActionHelper to assign tasks:
#       oh.needs_orders(airplane_obs)                                                   # gets untasked aircraft
#       oh.available_destinations(state, airplane_obs, plane_type: PlaneTypeID)         # Returns available destination from an airport node.
#       oh.get_lowest_cost_path(state, airport1, airport2, plane_type: PlaneTypeID)     # Gets the shortest path from airport1 to airport2 based on the plane model.

# from airlift.envs.airlift_env import ObservationHelper as oh

# from airlift.solutions.baselines import ShortestPath, RandomAgent



LOG_TO_CONSOLE = True
# LOG_TO_CONSOLE = False



class MySolution(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """

    def __init__(self):
        super().__init__()

        self._elapsed_steps = 0                         # manual frame counter

        self.cargo_items = None                         # dictionary of active cargo_items;  c.id is the key
        self.cargo_delivered = None
        self.prioratized_cargo_items = None             # sorted list of self.cargo_items; the sort order is hardest_deadline, then earliest pickup time
        self.cargo_delivery_path = None                 # dictionary: cargo_id : {'path', 'cost'}

        self.cost_routes_available = None               # plane type specific; flyable routes - can be flown now
        self.cost_routes_not_available = None           # plane type specific; flyable route currently not available
        
        self.plane_types = None                         # type of aircraft objects in the scenario
        self.airplane_id_list = None                    # aircraft ids
        self.aircraft = None                            # dictionary of aircraft agent objects
        
        self.cargo_item_to_aircraft_assignment = None   # a dictionary of tasks for each cargo item, aircraft and actions

        self.global_flight_path_graph = None
        
        self.full_delivery_paths = None
        self.plane_graph = None
        self.view = None
        self.global_view = None

        self.processing_time = 0                    # the time it takes to process an aircraft after it lands, prior to being tasked

        #Aircraft's states after arrival
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

        # #print cargo state info:
        # for c in state["active_cargo"]:
        #     print(f"\tDEBUG: cargo-item-{c.id}, earliest_pickup_time={c.earliest_pickup_time}, soft_deadline={c.soft_deadline}, hard_deadline={c.hard_deadline}")

        if LOG_TO_CONSOLE:
            print("\n\n-------mysolution.reset()-------")

        return




    def policies(self, obs, dones, infos):

        #clear path cache, ensures that cache lookup is only done once. This is necessary in case there are route malfunctions
        self.get_path_cost.cache_clear()

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

        # update active cargo
        self.get_cargo_items(state)

        #sorts cargo_items based on their hard_delivery time and earliest pickup time
        self.prioratize_cargo_items()

        # update cargo delivery paths
        self.calculate_cargo_delivery_paths()

        # task aircraft to move cargo
        actions = self.assign_actions_for_cargo()

        # update the time step
        self._elapsed_steps+=1
        # print(self._elapsed_steps)

        #Is this how you step through actions?  Do i need to this step, or its already handled when i return the list of action items?
        # env.step(actions)

        return actions

    def assign_actions_for_cargo(self):
        '''
            This method assigns aircraft actions to move cargo items to their destination. 

            aircraft states: MOVING → WAITING → PROCESSING → READY_FOR_TAKEOFF 
        '''

        
                            



        # task available aircraft to move to cargo

        #     if cargo is not at its final destination AND no aircraft scheduled to pick up this cargo item; -> need to task nearest capable aircraft.

        #         build the list of available aircraft and their distances to the cargo item

        #         choose the closest aircraft from list of available_aircraft

        #         we found at least one capable aircraft, task it
                
        
        #     task capable aircraft to cargo location
            
        
        # return actions
        return []

    def estimate_flight_time_for_flight_path (self,  flight_path: [int], plane: PlaneType):
        '''
            This method returns the estimated flight time for the given flight path and plane type.

            The flight path is a list of airports. AirportGenerator defines the processing time.

            flight_time = get_flight_time_between_directly_connected_airports(flight_distnace, plance_speed)
            estimated_flight_time = number_of_airports * processing_time + flight_time
        '''

        # processing_time is captured in the info->ScenarioObservation->processing_time=60)]
        number_of_airport_stops = len(flight_path)
        total_processing_time = number_of_airport_stops * self.processing_time
        
        # get total flight time
        total_flight_time = 0
        # for route in flight_path:
        #   self.cost_routes_available[plane_type][(airport_a, airport_b)]

        return total_processing_time + total_flight_time

    def get_flight_time_between_directly_connected_airports(self, distance, plane: PlaneType):
        '''
            This method calculates the time it takes to fly single route segment between two directly connected airports. 
            It is based on aircraft speed and distance between two airports.
            Note: this calculation is from route_generators.py / RouteGenerator._add_routes_at_random()
        '''

        return ceil(distance / plane.speed)

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
        Helper funciton that builds the look up tables for routes available (mal==0) and routes not available (mal>0)
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

    def get_cargo_items(self, state):
        '''
        gets a list of active cargo items.
        '''
        
        self.cargo_items = {}
        for c in state["active_cargo"]:
            self.cargo_items[c.id]=c

    # This method; returns a list of sorted cargo items. 
    def prioratize_cargo_items(self):
        '''
        Heper method that sorts cargo_items based on their hard_deadline, and then best on earliest pickup time
        '''
        self.prioratized_cargo_items = sorted(self.cargo_items.values(), key=lambda x: (x.hard_deadline, x.earliest_pickup_time))

        #need to get a list of cargo items that their deadline has not passed.

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
    
    # cached version of the the getpath
    @functools.lru_cache(maxsize=None)
    def get_path_cost(self, start: Airport, end: Airport, plane: PlaneType):
        '''
        Returns a summed cost of edges from start airport to end airport (using shortest path algorithm).
        (Cached version)
        '''
        # --- THE REST OF YOUR FUNCTION CODE REMAINS EXACTLY THE SAME ---
        if start == end:
            return {'path':[], 'cost': 0}

        elif start==NOAIRPORT_ID or end==NOAIRPORT_ID:
            return {'path':[], 'cost': float('inf')}

        path = []

        #get shortest path
        try:
            # NOTE: This relies on self.view[plane] being up-to-date
            path = nx.shortest_path(self.view[plane], start, end, weight="cost")[1:]
        except nx.NetworkXNoPath as e:
            # NOTE: This relies on self.global_view being up-to-date
            # print(e)
            try: # Added try/except for global path too, in case it also fails
                global_path = nx.shortest_path(self.global_view, start, end, weight="cost")[1:]
                if len(global_path) > 1:
                    # Recursive call - this will also use the cache if args are the same
                    res = self.get_path_cost(start, global_path[-2], plane)
                else: # Path exists but is only the start node, effectively no path
                    res = {'path':[], 'cost': float('inf')}
            except nx.NetworkXNoPath: # Global path also doesn't exist
                res = {'path':[], 'cost': float('inf')}

            # print(f"\n{e} Global_path={start}→{path}; best subpath={start}→{res['path']} cost={round(res['cost'], 3)}")
            return res

        #calculate cost of edges along the path
        cost = 0
        current_location = start # Renamed 'start_location' to avoid confusion with 'start' arg
        try:
            for p in path:
                # NOTE: This relies on self.cost_routes_available being up-to-date
                cost += self.cost_routes_available[plane][(current_location, p)]
                current_location = p
        except KeyError:
            # Handle cases where a path edge might not be in cost_routes_available
            # (e.g., if path finding and cost calculation data sources differ slightly)
            # This indicates a potential logic issue or data mismatch, but avoids crashing.
            # For performance, we assume the path found uses available routes.
            # If this happens frequently, investigate why.
            # Returning inf cost makes this path non-viable.
            return {'path':[], 'cost': float('inf')}


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
        digraph_obj = self.view[plane_type]
        pos = nx.spring_layout(digraph_obj)
        nx.draw(digraph_obj, pos, with_labels=True)

        #for display purpose only, round the edge costs
        rounded_labels = {}
        for k, v in self.cost_routes_available[plane_type].items():
            rounded_labels[k] = round(v, 3)
        
        nx.draw_networkx_edge_labels(digraph_obj, pos, edge_labels=rounded_labels)
        plt.show()

    def plot_airports_and_routes(self):

        #for display purpose only, round the edge costs
        rounded_labels = {}
        for k, v in self.edge_costs.items():
            rounded_labels[k] = round(v, 3)

        # visualize the above graph
        pos = nx.spring_layout(self.global_flight_path_graph)
        # pos = nx.fruchterman_reingold_layout(g)
        nx.draw(self.global_flight_path_graph, pos, with_labels=True)

        #lets trim the labels
        
        nx.draw_networkx_edge_labels(self.global_flight_path_graph, pos, edge_labels=self.rounded_labels)
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