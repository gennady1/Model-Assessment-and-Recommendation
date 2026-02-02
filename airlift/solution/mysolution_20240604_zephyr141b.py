import networkx as nx
import matplotlib.pyplot as plt


from pprint import pprint as pp
from functools import partial
from networkx import NetworkXNoPath


# from airlift.envs import ActionHelper
from airlift.solutions import Solution
from airlift.envs.agents import PlaneState
from airlift.envs.airport import Airport
from airlift.envs.plane_types import PlaneType
# from airlift.envs.route_map import RouteMap
from airlift.envs.airlift_env import ObservationHelper as oh, NOAIRPORT_ID 

# Should consider using ObservationHelper and ActionHelper to assign tasks
# from airlift.envs.airlift_env import ObservationHelper as oh

# LOG_TO_CONSOLE = True
LOG_TO_CONSOLE = False



class Heuristic_Zephyr141b(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """

    def __init__(self):
        super().__init__()

        self._elapsed_steps = 0                     #manually keeping track of time steps

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

        # print(f"\n\n  ***DEBUG***  \n\nobs:\n{obs} \n\ndones:\n{dones} \n\ninfos:\n{infos}")

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
        NOTE: This solution is an adaptation to the solution proposed by the zephyr:141b (LLM) (file: 20241205-Alg-design-zephyr-141b.txt)
            This method assigns aircraft actions to move cargo items to their destination. The heuristic is simple:

            Logic:
            1) check if aircraft is at airport (not moving)
                a) check if cargo has reached its destination. (two cases: final destination or the closest airport that aircraft is able to move it to)
                    -> Unload
                b) we have a cargo item(s) at the airport that needs to be moved, 
                    -> Load cargo if available and there is room on the aircraft
            2) move aircraft to cargo items. 
                -> assign and move aircraft to the highest priority cargo that is not yet scheduled
            3) do route updates, for the next location
                

            aircraft states: MOVING → WAITING → PROCESSING → READY_FOR_TAKEOFF 
        '''

        actions = {a: None for a in self.aircraft}
        
        # check aircraft at airport (actions: load or unload cargo items)
        for aid, a in self.aircraft.items():

            #initialize action list for each aircraft agent; if not initialized.
            actions[aid] = {"priority": 1,
                            "cargo_to_load": [],
                            "cargo_to_unload": [],
                            "destination": NOAIRPORT_ID}

            # 1) check if aircraft is at airport (not moving)
            if a['current_airport'] > 0:  #moving implies that aircraft is in flight

                for c in self.cargo_items.values():

                    # 1A check each on board cargo item if cargo has reached its destination.
                    if a['state'] != PlaneState.MOVING and c.id in a['cargo_onboard']:

                        #update cargo paths_to_destination if on board (cargo_item.location becomes 0)
                        path_cost = {}
                        for pt in self.plane_types:
                            path_cost[pt] = self.get_path_cost(a['current_airport'], c.destination, pt)
                        self.cargo_delivery_path[c.id] = path_cost
                    
                        # check if cargo has reached its final destination.
                        if a['current_airport'] == c.destination:

                            # small sanity check: dont task same action more than once.
                            if c.id in self.cargo_item_to_aircraft_assignment.keys():
                                self.cargo_delivered.append(c.id)
                                del self.cargo_item_to_aircraft_assignment[c.id]
                                actions[aid]['cargo_to_unload'].append(c.id)
                            
                            if LOG_TO_CONSOLE:
                                print(f"\nACTION (t={self._elapsed_steps}): FINAL DESTINATION; {aid}-UNLOAD; Cargo_Item-{c.id}; Airport-{a['current_airport']}; soft_dealine={c.soft_deadline}, hard_deadline={c.hard_deadline}")
                                # self.print_state_info()
                                # print("print Final Actions:")
                                # pp(actions)
                                # print("")

                        # not at final destination
                        else:
                            cargo_dest_path = self.cargo_delivery_path[c.id][a['plane_type']]['path']

                            # this aircraft is not able to take the cargo any closer toward final destination
                            if cargo_dest_path == []:

                                #same sanity check to make sure that we do not task same action it more than once
                                if c.id not in a['next_action']['cargo_to_unload']:
                                    if c.id in self.cargo_item_to_aircraft_assignment:
                                        del self.cargo_item_to_aircraft_assignment[c.id]
                                    
                                    actions[aid]['cargo_to_unload'].append(c.id)

                                    if LOG_TO_CONSOLE:
                                        print(f"\nACTION (t={self._elapsed_steps}): {aid}-UNLOAD; Cargo_Item-{c.id}; aircraft-route-limit-reached; Airport-{a['current_airport']}")
                                        # self.print_state_info()
                                        # print("print Final Actions:")
                                        # pp(actions)
                                        # print("")

                            else:  # update next destination for cargo enroute

                                if a['state'] == PlaneState.READY_FOR_TAKEOFF and a['next_action']['destination']==0:
                                    actions[aid]['destination'] = cargo_dest_path[0]
                                    actions[aid]['priority']+=1
                                    
                                    if LOG_TO_CONSOLE:
                                        print(f"\nACTION (t={self._elapsed_steps}): {aid}-CARGO_ONBOARD_NEXT_ROUTE; {a['current_airport']} → {cargo_dest_path[0]}; Cargo_Item-{c.id}")
                                        # self.print_state_info()
                                        # print("print Final Actions:")
                                        # pp(actions)
                                        # print("")
                
                    # 1b) we have cargo item(s) at the this airport that need to be moved AND avialble for pickup AND aircraft has capacity
                    elif c.location == a['current_airport'] and c.location != c.destination and c.is_available and c.weight + a['current_weight'] <= a['max_weight']:

                        # get cargo path route for this aircraft. check if it can move the cargo_item closer to the destination if not skip it
                        cargo_dest_path = self.cargo_delivery_path[c.id][a['plane_type']]['path']
                        
                        #  next hop on the cargo path is reachable by this aircraft AND not
                        # optimizaiton option here: possibly swap more important item; heavier item
                        if len(cargo_dest_path)>0 and cargo_dest_path[0] in a['available_routes'] and not c.id in a['next_action']['cargo_to_load']:

                            # #same sanity check - dont add if already tasked
                           
                            self.cargo_item_to_aircraft_assignment[c.id] = aid              #register cargo to aircraft assignment
                            actions[aid]['cargo_to_load'].append(c.id)
                            actions[aid]['destination'] = cargo_dest_path[0]

                            if LOG_TO_CONSOLE:
                                print(f"\nACTION (t={self._elapsed_steps}): {aid}-LOAD_CARGO & SET_DESTINATION; Cargo_Item-{c.id}; {a['current_airport']} → {cargo_dest_path[0]}")
                                # pp(a['next_action'])
                            



        # task available aircraft to move to cargo
        for pci in self.prioratized_cargo_items:

            # if cargo is not at its final destination AND no aircraft scheduled to pick up this cargo item; -> need to task nearest capable aircraft.
            if pci.location != pci.destination and pci.id not in self.cargo_item_to_aircraft_assignment.keys():

                # build the list of available aircraft and their distances to the cargo item
                available_aircraft = {}
                for aid, a in self.aircraft.items():

                    # can aircraft reach cargo location?
                    path_cost_to_cargo = self.get_path_cost(a['current_airport'], pci.location, a['plane_type'])
                    can_reach_cargo = self.aircraft[aid]['state'] == PlaneState.READY_FOR_TAKEOFF and len(path_cost_to_cargo['path'])>0 and path_cost_to_cargo['path'][-1]==pci.location

                    # can aircraft move cargo at least one hop closer to final destination; more of an opportunitsitc 
                    path_cost_to_cargo_dest = self.get_path_cost(pci.location, pci.destination, a['plane_type'])
                    can_move_cargo_closer_to_dest = self.aircraft[aid]['state'] == PlaneState.READY_FOR_TAKEOFF and len(path_cost_to_cargo_dest['path'])>0

                    # check if this is capable aircraft to task and cargo_item is not loaded
                    if can_reach_cargo and can_move_cargo_closer_to_dest and pci.id not in a['cargo_onboard']:

                        # aircraft has no scheduled actions
                        if actions[aid]['cargo_to_load'] == []  and actions[aid]['cargo_to_unload'] == [] and actions[aid]['destination'] == NOAIRPORT_ID and pci.weight + a['current_weight'] <= a['max_weight']:
                            available_aircraft[aid] = path_cost_to_cargo

                # choose the closest aircraft from list of available_aircraft
                aircraft_id = None
                lowest_cost = None
                path_to_cargo = None
                for aid, path_cost in available_aircraft.items():
                    if lowest_cost is None:
                        aircraft_id = aid
                        lowest_cost = path_cost['cost']
                        path_to_cargo = path_cost['path']
                    elif path_cost['cost'] < lowest_cost:
                        aircraft_id = aid
                        lowest_cost = path_cost['cost']
                        path_to_cargo = path_cost['path']

                # we found at least one capable aircraft, task it
                if aircraft_id is not None and actions[aircraft_id]['destination']==0 and aircraft_id not in self.cargo_item_to_aircraft_assignment.values() and len(path_to_cargo)>0:

                    # potential optimization: it might make sense to first check already tasked aircraft should complete existing delivery;
                    self.cargo_item_to_aircraft_assignment[pci.id] = aircraft_id
                    actions[aircraft_id]['destination'] = path_to_cargo[0]

                    if LOG_TO_CONSOLE:
                        print(f"\nACTION (t={self._elapsed_steps}): {aircraft_id}-MOVE: {self.aircraft[aircraft_id]['current_airport']} → {path_to_cargo}; Pick Cargo_Item-{pci.id} @ airport {pci.location}")
                        # self.print_state_info()
                        # print("\nDEBUG Final Actions:")
                        # pp(actions)
                        # print("")
            

            #task capable aircraft to cargo location
            for cargo_id, aircraft_id in self.cargo_item_to_aircraft_assignment.items():
                if cargo_id in self.cargo_items.keys():
                    aloc = self.aircraft[aircraft_id]['current_airport']
                    cloc = self.cargo_items[cargo_id].location

                    path_to_cargo = self.get_path_cost(aloc, cloc, self.aircraft[aircraft_id]['plane_type'])
                    if self.aircraft[aircraft_id]['state'] == PlaneState.READY_FOR_TAKEOFF and actions[aircraft_id]['destination'] == NOAIRPORT_ID  and len(path_to_cargo['path'])>0 and self.aircraft[aircraft_id]['current_airport'] != cloc:

                        actions[aircraft_id]['destination'] = path_to_cargo['path'][0]
                        
                        if LOG_TO_CONSOLE:
                            print(f"\nACTION (t={self._elapsed_steps}): {aircraft_id}-NEXT_ROUTE; {self.aircraft[aircraft_id]['current_airport']} -> {path_to_cargo['path'][0]}; Cargo_Item-{cargo_id}")
                            # self.print_state_info()
                            # print("\nDEBUG Final Actions:")
                            # pp(actions)
                            # print("")
                else:
                    print(f"\n\tcargo_item Lookup Failure, cargo_id={cargo_id} not in dictionary self.cargo_items.keys()={self.cargo_items.keys()}")
        
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

    def get_cargo_items(self, state):
        '''
        gets a list of active cargo items.
        '''
        
        self.cargo_items = {}
        for c in state["active_cargo"]:
            self.cargo_items[c.id]=c

    def prioratize_cargo_items(self):
        '''
        Simple heper method that sorts cargo_items based on their hard_deadline, and then best on earliest pickup time
        '''
        self.prioratized_cargo_items = sorted(self.cargo_items.values(), key=lambda x: (x.hard_deadline, x.earliest_pickup_time))

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
        digraph_obj = self.view[plane_type]
        pos = nx.spring_layout(digraph_obj)
        nx.draw(digraph_obj, pos, with_labels=True)

        #for display purpose only, round the edge costs
        rounded_labels = {}
        for k, v in self.cost_routes_available[plane_type].items():
            rounded_labels[k] = round(v, 3)
        
        nx.draw_networkx_edge_labels(digraph_obj, pos, edge_labels=rounded_labels)
        plt.show()

    # def plot_airports_and_routes(self):

    #     #for display purpose only, round the edge costs
    #     rounded_labels = {}
    #     for k, v in self.edge_costs.items():
    #         rounded_labels[k] = round(v, 3)

    #     # visualize the above graph
    #     pos = nx.spring_layout(self.global_flight_path_graph)
    #     # pos = nx.fruchterman_reingold_layout(g)
    #     nx.draw(self.global_flight_path_graph, pos, with_labels=True)

    #     #lets trim the labels
        
    #     nx.draw_networkx_edge_labels(self.global_flight_path_graph, pos, edge_labels=self.rounded_labels)
    #     plt.show()

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

    def plot_airports_and_routes(self):

        # visualize the above graph
        pos = nx.spring_layout(self.global_flight_path_graph)
        # pos = nx.fruchterman_reingold_layout(g)
        nx.draw(self.global_flight_path_graph, pos, with_labels=True)

        #for display purpose only, round the edge costs
        rounded_labels = {}
        for plane_type in self.plane_types:
            for k, v in self.cost_routes_available[plane_type].items():
                if k in rounded_labels:
                    rounded_labels[k] = str(plane_type) + ', ' + rounded_labels[k]
                else:
                    rounded_labels[k] = str(plane_type) + ': ' + str(round(v, 3))
        
        nx.draw_networkx_edge_labels(self.global_flight_path_graph, pos, edge_labels=rounded_labels)
        plt.show()
        