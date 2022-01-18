import pickle
import logging

from abc import ABC, abstractmethod
import os
from typing import Optional, Tuple

import yaml
from Utils.DB.Client import Client, SaGeClient
import numpy as np

config = yaml.load(open(os.environ["RTOS_CONFIG"], 'r'), Loader=yaml.FullLoader)[os.environ["RTOS_TRAINTYPE"]]

class Rewarder(ABC):
    """Class description"""

    def __init__(self, client: Client):
        self._client = client
        self._plans = dict()
        self._costs = dict()
        self._progressions = dict()

    @staticmethod
    @abstractmethod
    def get_type() -> str:
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        pass

    @property
    def client(self) -> Client:
        return self._client

    def get_query_plan(self, query: str) -> str:
        join_order = hash(query)
        if join_order not in self._plans:
            self._plans[join_order] = query
        return self._plans[join_order]

    @abstractmethod
    def get_cost(self, query: str) -> float:
        join_order = hash(query)
        if join_order not in self._costs:
            self._costs[join_order] = self.client.query_cost(query, force_order=True)
        return self._costs[join_order]

    def compute_refined_cost(
        self, query: str, next: Optional[str] = None, force_order: bool = True
    ) -> Tuple[Optional[str], float]:
        response = self.client.execute_query(
            query, next=next, force_order=force_order
        )
        return response['next'], response['stats']['metrics']['cost']

    def get_refined_cost(self, query: str) -> float:
        join_order = hash(query)
        plan = self.get_query_plan(query)

        # initialize data structures
        if join_order not in self._progressions:
            self._progressions[join_order] = None
        if join_order not in self._costs:
            self._costs[join_order] = -1

        # the join order has already been fully executed
        if self._costs[join_order] >= 0:
            return self._costs[join_order]

        # refine the join order's cost one quantum more
        next = self._progressions[join_order]
        next, cost = self.compute_refined_cost(
            plan, next=next, force_order=True
        )
        self._progressions[join_order] = next

        return cost

    @abstractmethod
    def get_reward(self, query: str) -> float:
        pass

class SaGeRefinedCostImprovementRewarder(Rewarder):
    """Class description"""

    @staticmethod
    def create():
        fn = os.path.join("models", config["model"]["name"], "SaGeRefinedCostImprovementRewarder.pkl")
        if os.path.exists(fn):
            return pickle.load(open(fn, mode="rb"))
        return SaGeRefinedCostImprovementRewarder()

    def save(self):
        fn = os.path.join("models", config["model"]["name"], "SaGeRefinedCostImprovementRewarder.pkl")
        pickle.dump(self, open(fn, mode="wb"))

    def __init__(self):
        super().__init__(
            SaGeClient(
                endpoint=f'http://{config["database"]["sage_host"]}:{config["database"]["sage_port"]}/{config["database"]["sage_endpoint"]}',
                graph=config["database"]["sage_graph"]
            )
        )
        self._refinements = dict()

    @staticmethod
    def get_type() -> str:
        return 'sage-refined-cost-improvement'

    @property
    def type(self) -> str:
        return 'sage-refined-cost-improvement'

    def get_baseline_cost(self, query: str) -> float:
        # initialize data structures
        queryHash = hash(query)
        if hash(queryHash) not in self._costs:
            self._costs[queryHash] = -1

        # the baseline's cost doesn't need to be refined
        old_refinement = self._refinements[f'{queryHash}/old']
        new_refinement = self._refinements[f'{queryHash}/new']
        if new_refinement <= old_refinement:
            return self._costs[queryHash]

        # refine the baseline's cost one quantum more
        if queryHash not in self._progressions:
            next = None
        elif self._progressions[queryHash] is not None:
            next = self._progressions[queryHash]
        else:
            return self._costs[queryHash]

        next, cost = self.compute_refined_cost(
            query, next=next, force_order=False
        )

        self._costs[queryHash] = cost
        self._progressions[queryHash] = next
        self._refinements[f'{queryHash}/old'] = new_refinement

        return cost

    def get_cost(self, query: str) -> float:
        queryHash = hash(query)

        # initialize data structures
        if f'{queryHash}/old' not in self._refinements:
            self._refinements[f'{queryHash}/old'] = 0
        if f'{queryHash}/new' not in self._refinements:
            self._refinements[f'{queryHash}/new'] = 0
        if queryHash not in self._refinements:
            self._refinements[queryHash] = 0

        cost = self.get_refined_cost(query)
        self.save()

        # update the join order's refinement level
        refinement = self._refinements[queryHash] + 1
        if refinement > self._refinements[f'{queryHash}/old']:
            self._refinements[f'{queryHash}/new'] = refinement
        self._refinements[queryHash] = refinement

        return cost
    
    def get_reward(self, query: str) -> Tuple[float, float]:
        cost = self.get_cost(query) + 1
        baseline_cost = self.get_baseline_cost(query) + 1
        #reward = min(max(np.log10(cost / baseline_cost, 10), -10), 10)
        return cost, baseline_cost
