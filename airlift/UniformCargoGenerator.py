from typing import List, Collection, NamedTuple, Tuple
from airlift.envs.generators.cargo_generators import StaticCargoGenerator
from airlift.envs.route_map import RouteMap
from airlift.envs.cargo import Cargo, CargoID
from ordered_set import OrderedSet
from airlift.envs.airport import Airport, AirportID
import networkx as nx

class UniformCargoGenerator(StaticCargoGenerator):
    """
    Handles the generation of static cargo tasks (using uniform distribution) These are initialized upon the creation of the environment.
    """

    def __init__(self, num_of_tasks=1, soft_deadline_multiplier=50, hard_deadline_multiplier=100, max_stagger_steps=100, max_weight=1, max_cycles=1440, earliest_pickup_offset=200, hard_deadline_offset=200):
        super().__init__(num_of_tasks, num_of_tasks, max_weight=max_weight)

        self.max_stagger_steps = max_stagger_steps
        self._processing_time = None
        self._graph = None
        self.soft_deadline_multiplier = soft_deadline_multiplier
        self.hard_deadline_multiplier = hard_deadline_multiplier
        self.avg_hops = None
        self.avg_flighttime = None
        self.max_cycles = max_cycles
        self.earliest_pickup_offset = earliest_pickup_offset
        self.hard_deadline_offset = hard_deadline_offset

    def reset(self, routemap: RouteMap):
        super().reset(routemap)
        self._processing_time = routemap.airports[0].processing_time  # Assume all processing times are the same

        self.avg_hops = nx.average_shortest_path_length(routemap.multigraph, weight=None) - 1
        self.avg_flighttime = nx.average_shortest_path_length(routemap.multigraph, weight="time")

    def generate_initial_orders(self) -> list:
        """
        Generates static cargo orders upon creation of the environment.

        :return: `order_list` - Returns a set that contains all the (static) generated cargo orders

        """

        self.current_cargo_count = self.num_initial_tasks
        cargo_list = []
        for i in range(self.num_initial_tasks):
            if self.max_stagger_steps != 0:
                stagger_duration = self._np_random.integers(0, self.max_stagger_steps)
            else:
                stagger_duration = self.max_stagger_steps
            cargo_list.append(self.generate_cargo_order(i, self.routemap.drop_off_airports, self.routemap.pick_up_airports, time_available=stagger_duration))
        return cargo_list

    def generate_dynamic_orders(self, elapsed_steps, max_cycles) -> List[Cargo]:
        return []

    def generate_cargo_order(self, cargo_id, drop_off_airports: OrderedSet[Airport],
                             pickup_airports: OrderedSet[Airport], time_available=0,
                             soft_deadline_multiplier=None, hard_deadline_multiplier=None) -> Cargo:
        """
        Generates cargo orders based on several parameters. Takes into account if we are running a scenario with a
        concentrated drop off or pick up location as well as non-concentrated locations that utilize the entire map.
        The dynamic cargo generator accesses this function without going through the above generate_order function.

        :parameter cargo_id: Incremental count of what cargo to create
        :parameter drop_off_airports: List that contains all the airports that are drop off locations'
        :parameter pick_up_airports: List that contains all the airports that are pick up locations
        :return: `cargo_task` : A fully generated cargo task

        """

        #choose (raindom) drop off and pick up airports
        destination_airport = self._np_random.choice(drop_off_airports)
        source_airport = self._np_random.choice(pickup_airports - {destination_airport})

        # set earliest earliest available pick up time and soft, and hard deadlines for cargo item
        soft_deadline = (self.max_cycles / self.num_initial_tasks) * cargo_id


        if soft_deadline == 0:
            soft_deadline = 1

        # sanity check, if soft deadline is greater than the maximum number of cycles, set it to be equal to the maximum number of cycles minus the hard_deadline_offset
        if soft_deadline > self.max_cycles:
            soft_deadline = self.max_cycles-self.hard_deadline_offset

        hard_deadline = min(self.max_cycles, soft_deadline + self.hard_deadline_offset)
        time_available = max(1, soft_deadline - self.earliest_pickup_offset)

        # print(f"\nDEBUG: Cargo-Item-Schedule: time_available={time_available}, soft_deadline={soft_deadline}, hard_deadline={hard_deadline}")

        assert soft_deadline > 0
        assert hard_deadline > 0
        cargo_task = Cargo(cargo_id,
                           source_airport,
                           destination_airport,
                           self.generate_cargo_weight(),
                           soft_deadline,
                           hard_deadline,
                           earliest_pickup_time=time_available)
        source_airport.add_cargo(cargo_task)

        return cargo_task