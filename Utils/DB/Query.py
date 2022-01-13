import os
from typing import List, Tuple, Dict, Any, Optional
from sage.query_engine.iterators.loader import SavedProtobufPlan
from sage.query_engine.protobuf.utils import pyDict_to_protoDict
from sage.query_engine.protobuf.iterators_pb2 import (
    SavedValuesIterator, SavedScanIterator, SavedIndexJoinIterator,
    SavedFilterIterator
)
import yaml

from Utils.Parser.parsed_query import ParsedQuery

from typing import List
from sage.query_engine.iterators.loader import SavedProtobufPlan

from typing import Optional, Tuple, Set, List, Dict, Any
from datetime import datetime

from sage.query_engine.optimizer.logical.visitors.pipeline_builder import ExpressionStringifier
from sage.query_engine.optimizer.physical.visitors.filter_push_down import FilterPushDown
from sage.query_engine.optimizer.physical.visitors.query_plan_stringifier import QueryPlanStringifier
from sage.query_engine.iterators.preemptable_iterator import PreemptableIterator
from sage.query_engine.iterators.loader import SavedProtobufPlan
from sage.query_engine.iterators.projection import ProjectionIterator
from sage.query_engine.iterators.scan import ScanIterator
from sage.query_engine.iterators.values import ValuesIterator
from sage.query_engine.iterators.nlj import IndexJoinIterator
from sage.query_engine.iterators.filter import FilterIterator
from sage.database.backends.db_connector import DatabaseConnector
from sage.database.backends.db_iterator import DBIterator
from sage.query_engine.protobuf.iterators_pb2 import RootTree

from typing import List
from sage.query_engine.iterators.loader import SavedProtobufPlan
from sage.query_engine.protobuf.iterators_pb2 import (
    SavedFilterIterator, SavedValuesIterator, SavedIndexJoinIterator,
    SavedProjectionIterator, SavedScanIterator, RootTree
)

config = yaml.load(open(os.environ["RTOS_CONFIG"], 'r'), Loader=yaml.FullLoader)[os.environ["RTOS_TRAINTYPE"]]

class SavedPlanFlattener():

    def visit(self, node: SavedProtobufPlan) -> List[SavedProtobufPlan]:
        if type(node) is RootTree:
            return self.visit_root(node)
        if type(node) is SavedFilterIterator:
            return self.visit_filter(node)
        if type(node) is SavedProjectionIterator:
            return self.visit_projection(node)
        elif type(node) is SavedScanIterator:
            return self.visit_scan(node)
        elif type(node) is SavedIndexJoinIterator:
            return self.visit_join(node)
        elif type(node) is SavedValuesIterator:
            return self.visit_values(node)
        else:
            raise Exception(f"Unknown iterator {type(node)}")

    def visit_root(self, node: RootTree) -> List[SavedProtobufPlan]:
        child = getattr(node, node.WhichOneof('source'))
        return self.visit(child)

    def visit_projection(
        self, node: SavedProjectionIterator
    ) -> List[SavedProtobufPlan]:
        child = getattr(node, node.WhichOneof('source'))
        return self.visit(child)

    def visit_filter(
        self, node: SavedFilterIterator
    ) -> List[SavedProtobufPlan]:
        child = getattr(node, node.WhichOneof('source'))
        flattened_plan = self.visit(child)
        flattened_plan.append(node)
        return flattened_plan

    def visit_join(
        self, node: SavedIndexJoinIterator
    ) -> List[SavedProtobufPlan]:
        left_child = getattr(node, node.WhichOneof('left'))
        right_child = getattr(node, node.WhichOneof('right'))
        flattened_plan = self.visit(left_child)
        flattened_plan.extend(self.visit(right_child))
        flattened_plan.append(node)
        return flattened_plan

    def visit_values(
        self, node: SavedValuesIterator
    ) -> List[SavedProtobufPlan]:
        return [node]

    def visit_scan(self, node: SavedScanIterator) -> List[SavedProtobufPlan]:
        return [node]


class DummyIterator(DBIterator):

    def last_read(self) -> Optional[str]:
        return None

    def next(self) -> Optional[str]:
        return None


class DummyConnector(DatabaseConnector):

    def search(
        self, subject: str, predicate: str, obj: str,
        last_read: Optional[str] = None,
        as_of: Optional[datetime] = None
    ) -> Tuple[DBIterator, int]:
        return DummyIterator({}), -1

    def from_config(config: dict):
        """Build a DatabaseConnector from a dictionnary"""
        pass


class QueryPlanBuilder():

    def __collect_join_variables__(
        self, join_order: List[int], parsed_query: ParsedQuery
    ) -> Set[str]:
        variables = set()
        for i in range(len(join_order)):
            triple_pattern = parsed_query.get_triple_pattern(join_order[i])
            if triple_pattern['subject'].startswith('?'):
                variables.add(triple_pattern['subject'])
            if triple_pattern['object'].startswith('?'):
                variables.add(triple_pattern['object'])
        return variables

    def __is_values_defined__(
        self, values: List[Dict[str, str]], variables: Set[str]
    ) -> bool:
        values_variables = set(values[0].keys())
        return variables.issuperset(values_variables)

    def __create_values_iterators__(
        self, join_order: List[int], parsed_query: ParsedQuery,
        variables: Set[str]
    ) -> Optional[PreemptableIterator]:
        if len(parsed_query.values) == 0:
            return None
        pipeline = None
        for values in parsed_query.values:
            if self.__is_values_defined__(values, variables):
                iterator = ValuesIterator(values)
                if pipeline is None:
                    pipeline = iterator
                else:
                    pipeline = IndexJoinIterator(pipeline, iterator)
        return pipeline

    def __create_scan_iterators__(
        self, pipeline: Optional[PreemptableIterator], join_order: List[int],
        parsed_query: ParsedQuery
    ) -> PreemptableIterator:
        index = 0
        if pipeline is None:
            triple_pattern = parsed_query.get_triple_pattern(join_order[0])
            pipeline = ScanIterator(DummyConnector(), triple_pattern)
            index = 1
        while index < len(join_order):
            triple_pattern = parsed_query.get_triple_pattern(
                join_order[index]
            )
            pipeline = IndexJoinIterator(
                pipeline, ScanIterator(DummyConnector(), triple_pattern)
            )
            index += 1
        return pipeline

    def __is_filter_defined__(self, filter: Any, variables: Set[str]) -> bool:
        filter_variables = set(filter.vars)
        return variables.issuperset(filter_variables)

    def __create_filter_iterators__(
        self, pipeline: PreemptableIterator, join_order: List[int],
        parsed_query: ParsedQuery, variables: Set[str]
    ) -> PreemptableIterator:
        for filter in parsed_query.filters:
            if self.__is_filter_defined__(filter, variables):
                raw_expression = ExpressionStringifier().visit(filter)
                pipeline = FilterIterator(pipeline, raw_expression, filter)
        return pipeline

    def build(
        self, join_order: List[int], parsed_query: ParsedQuery
    ) -> PreemptableIterator:
        variables = self.__collect_join_variables__(
            join_order, parsed_query
        )
        pipeline = self.__create_values_iterators__(
            join_order, parsed_query, variables
        )
        pipeline = self.__create_scan_iterators__(
            pipeline, join_order, parsed_query
        )
        pipeline = self.__create_filter_iterators__(
            pipeline, join_order, parsed_query, variables
        )
        pipeline = ProjectionIterator(pipeline, parsed_query.projection)
        return FilterPushDown().visit(pipeline)

    def save(self, query_plan: PreemptableIterator) -> SavedProtobufPlan:
        saved_plan = RootTree()
        source_field = f'{query_plan.serialized_name()}_source'
        getattr(saved_plan, source_field).CopyFrom(query_plan.save())
        return saved_plan

    def stringify(
        self, join_order: List[int], parsed_query: ParsedQuery
    ) -> str:
        pipeline = self.build(join_order, parsed_query)
        return QueryPlanStringifier().visit(pipeline)


class SuspendedQuery():

    def __init__(self, saved_plan: SavedProtobufPlan):
        self._saved_plan = saved_plan
        self._flattened_saved_plan = SavedPlanFlattener().visit(saved_plan)

    @property
    def saved_plan(self) -> SavedProtobufPlan:
        return self._saved_plan

    @property
    def flattened_saved_plan(self) -> List[SavedProtobufPlan]:
        return self._flattened_saved_plan


class QueryTracker():

    def __init__(self):
        self._suspended_queries = dict()
        self._last_updated = None

    def same_relation(
        self, left: SuspendedQuery, right: SuspendedQuery, position: int
    ) -> bool:
        index = position
        while index < len(left.flattened_saved_plan):
            left_node = left.flattened_saved_plan[index]
            right_node = right.flattened_saved_plan[index]
            if type(left_node) != type(right_node):
                return False
            elif type(left_node) == SavedScanIterator:
                return left_node.pattern == right_node.pattern
            elif type(left_node) == SavedValuesIterator:
                return left_node.values == right_node.values
            index += 1
        return True

    # def same_relation(
    #     self, left: SuspendedQuery, right: SuspendedQuery, index: int
    # ) -> bool:
    #     left_node_type = type(left.flattened_saved_plan[index])
    #     right_node_type = type(right.flattened_saved_plan[index])
    #     if left_node_type != right_node_type:
    #         return False
    #     elif left_node_type == SavedScanIterator:
    #         left_pattern = left.flattened_saved_plan[index].pattern
    #         right_pattern = right.flattened_saved_plan[index].pattern
    #         return left_pattern == right_pattern
    #     elif left_node_type == SavedValuesIterator:
    #         left_values = left.flattened_saved_plan[index].values
    #         right_values = right.flattened_saved_plan[index].values
    #         return left_values == right_values
    #     else:  # shound not be called with something else than VALUES/SCANS
    #         raise Exception('This function shound not be called with NLJ...')

    def most_advanced(
        self, left: SuspendedQuery, right: SuspendedQuery,
        prefix: SuspendedQuery
    ) -> int:
        index = 0
        while index < len(left.flattened_saved_plan):
            left_node = left.flattened_saved_plan[index]
            right_node = right.flattened_saved_plan[index]
            if type(left_node) == SavedIndexJoinIterator:
                index += 1
                continue
            if type(left_node) == SavedFilterIterator:
                index += 1
                continue
            same_left = self.same_relation(prefix, left, index)
            same_right = self.same_relation(prefix, right, index)
            if same_left and same_right:
                next_same_left = self.same_relation(prefix, left, index + 1)
                next_same_right = self.same_relation(prefix, right, index + 1)
                if next_same_left and next_same_right:
                    if left_node.produced == right_node.produced:
                        index += 1
                        continue
                    return left_node.produced > right_node.produced
                elif next_same_left and not next_same_right:
                    return left_node.produced >= right_node.produced - 1
                elif not next_same_left and next_same_right:
                    return left_node.produced - 1 > right_node.produced
                else:
                    return left_node.produced >= right_node.produced
            elif same_left and not same_right:
                return True
            elif not same_left and same_right:
                return False
            else:
                raise Exception('Both plans have no prefix in common...')

    # def most_advanced(
    #     self, left: SuspendedQuery, right: SuspendedQuery,
    #     prefix: SuspendedQuery
    # ) -> int:
    #     for i in range(len(prefix.flattened_saved_plan)):
    #         if (i > 0) and (i % 2) == 0:
    #             continue
    #         left_node = left.flattened_saved_plan[i]
    #         right_node = right.flattened_saved_plan[i]
    #         same_left = self.same_relation(prefix, left, i)
    #         same_right = self.same_relation(prefix, right, i)
    #         if same_left and same_right:
    #             next = 1 if i == 0 else i + 2
    #             if next >= len(prefix.flattened_saved_plan):
    #                 return left_node.produced >= right_node.produced
    #             same_left_next = self.same_relation(prefix, left, next)
    #             same_right_next = self.same_relation(prefix, right, next)
    #             if same_left_next and same_right_next:
    #                 if left_node.produced == right_node.produced:
    #                     continue
    #                 return left_node.produced > right_node.produced
    #             elif same_left_next and not same_right_next:
    #                 return left_node.produced >= right_node.produced - 1
    #             elif not same_left_next and same_right_next:
    #                 return left_node.produced - 1 > right_node.produced
    #             else:
    #                 return left_node.produced >= right_node.produced
    #         elif same_left and not same_right:
    #             return True
    #         elif not same_left and same_right:
    #             return False
    #         else:
    #             raise Exception('Both plans have no prefix in common...')

    def print_plan(self, plan: SuspendedQuery) -> None:
        repr = []
        for node in plan.flattened_saved_plan:
            if type(node) == SavedScanIterator:
                repr += [node.pattern.predicate]
            elif type(node) == SavedValuesIterator:
                repr += ['V']
            elif type(node) == SavedFilterIterator:
                repr += ['F']
            elif type(node) == SavedIndexJoinIterator:
                repr += ['J']
        print(f"plan: {' - '.join(repr)}")

    def share_progression(
        self, source: SuspendedQuery, target: SuspendedQuery
    ) -> None:
        index = 0
        # self.print_plan(source)
        # print('')
        # self.print_plan(target)
        while index < len(source.flattened_saved_plan):
            source_node = source.flattened_saved_plan[index]
            target_node = target.flattened_saved_plan[index]
            # print(f'index: {index} - type: {type(source_node)}')
            # print(f'index: {index} - type: {type(target_node)}')
            if type(source_node) != type(target_node):
                # print('breaking because different')
                break
            elif type(source_node) == SavedIndexJoinIterator:
                pyDict_to_protoDict(source_node.muc, target_node.muc)
            elif type(source_node) == SavedFilterIterator:
                target_node.consumed = source_node.consumed
                target_node.produced = source_node.produced
                # print(f'source: {source_node.expression}, consumed: {source_node.consumed}, produced: {source_node.produced}')
                # print(f'target: {target_node.expression}, consumed: {target_node.consumed}, produced: {target_node.produced}')
            elif not self.same_relation(source, target, index):
                # print('breaking because not same relation')
                break
            elif type(source_node) == SavedValuesIterator:
                # print(f'source: {source_node.values}')
                # print(f'target: {target_node.values}')
                pyDict_to_protoDict(source_node.muc, target_node.muc)
                target_node.next_value = source_node.next_value
                target_node.produced = source_node.produced
                if not self.same_relation(source, target, index + 1):
                    if target_node.next_value > 0:
                        target_node.next_value -= 1
                        target_node.produced -= 1
            elif type(source_node) == SavedScanIterator:
                # print(f'source: {source_node.pattern}')
                # print(f'target: {target_node.pattern}')
                pyDict_to_protoDict(source_node.mu, target_node.mu)
                pyDict_to_protoDict(source_node.muc, target_node.muc)
                target_node.last_read = source_node.last_read
                target_node.produced = source_node.produced
                target_node.pattern_produced = source_node.pattern_produced
                target_node.cumulative_cardinality = source_node.cumulative_cardinality
                target_node.pattern_cardinality = source_node.pattern_cardinality
                target_node.stages = source_node.stages
                last_read = int(source_node.last_read)
                if not self.same_relation(source, target, index + 1):
                    if len(source_node.mu) == 0 and last_read > 0:
                        target_node.last_read = str(last_read - 1)
                        target_node.produced -= 1
                        target_node.pattern_produced -= 1
            else:
                raise Exception('Unexpected iterator in the flattened plan...')
            index += 1
        # print(f'last: {index}')

    # def share_progression(
    #     self, source: SuspendedQuery, target: SuspendedQuery
    # ) -> None:
    #     for i in range(len(source.flattened_saved_plan)):
    #         source_node = source.flattened_saved_plan[i]
    #         target_node = target.flattened_saved_plan[i]
    #         if (i > 0) and (i % 2) == 0:
    #             pyDict_to_protoDict(source_node.muc, target_node.muc)
    #             continue
    #         if not self.same_relation(source, target, i):
    #             break
    #         next = 1 if i == 0 else i + 2
    #         if type(source_node) == SavedValuesIterator:
    #             pyDict_to_protoDict(source_node.muc, target_node.muc)
    #             target_node.next_value = source_node.next_value
    #             target_node.produced = source_node.produced
    #             if next >= len(source.flattened_saved_plan):
    #                 continue
    #             elif not self.same_relation(source, target, next):
    #                 if target_node.next_value > 0:
    #                     target_node.next_value -= 1
    #                     target_node.produced -= 1
    #         else:
    #             pyDict_to_protoDict(source_node.mu, target_node.mu)
    #             pyDict_to_protoDict(source_node.muc, target_node.muc)
    #             target_node.last_read = source_node.last_read
    #             target_node.produced = source_node.produced
    #             target_node.pattern_produced = source_node.pattern_produced
    #             target_node.cumulative_cardinality = source_node.cumulative_cardinality
    #             target_node.pattern_cardinality = source_node.pattern_cardinality
    #             target_node.stages = source_node.stages
    #             last_read = int(source_node.last_read)
    #             if next >= len(source.flattened_saved_plan):
    #                 continue
    #             elif not self.same_relation(source, target, next):
    #                 if len(source_node.mu) == 0 and last_read > 0:
    #                     target_node.last_read = str(last_read - 1)
    #                     target_node.produced -= 1
    #                     target_node.pattern_produced -= 1

    def search_most_advanced_query(
        self, join_order: List[int], new_query_plan: SuspendedQuery
    ) -> SuspendedQuery:
        join_order = str(join_order)
        if join_order in self._suspended_queries:
            most_advanced_query = self._suspended_queries[join_order]
        else:
            most_advanced_query = None
        for suspended_query in self._suspended_queries.values():
            if most_advanced_query is None:
                if self.same_relation(new_query_plan, suspended_query, 0):
                    most_advanced_query = suspended_query
            elif self.most_advanced(
                suspended_query, most_advanced_query, new_query_plan
            ):
                most_advanced_query = suspended_query
        return most_advanced_query

    def report_progression(
        self, join_order: List[int], saved_plan: SavedProtobufPlan
    ) -> None:
        if saved_plan is not None:
            join_order = str(join_order)
            suspended_query = SuspendedQuery(saved_plan)
            self._suspended_queries[join_order] = suspended_query
            self._last_updated = join_order

    def get_progression(
        self, join_order: List[int]
    ) -> Optional[SavedProtobufPlan]:
        join_order = str(join_order)
        if join_order in self._suspended_queries:
            return self._suspended_queries[join_order].saved_plan
        return None

    def get_last_saved_plan(self) -> Optional[SavedProtobufPlan]:
        if self._last_updated is None:
            return None
        return self._suspended_queries[self._last_updated].saved_plan

    def restore_progression(
        self, join_order: List[int], parsed_query: ParsedQuery
    ) -> SavedProtobufPlan:
        plan_builder = QueryPlanBuilder()
        query_plan = plan_builder.build(join_order, parsed_query)
        saved_plan = plan_builder.save(query_plan)
        new_query_plan = SuspendedQuery(saved_plan)
        most_advanced_query = self.search_most_advanced_query(
            join_order, new_query_plan
        )
        if most_advanced_query is not None:
            self.share_progression(most_advanced_query, new_query_plan)
        return new_query_plan.saved_plan

    def reset(self) -> None:
        self._suspended_queries = dict()
        self._last_updated = None

class Query:
    def __init__(self, runner = None, query: str = None, filename: str = None):
        self.DPLantency = None
        self.DPCost = None
        self.bestLatency = None
        self.bestCost = None
        self.bestOrder = None
        self.plTime = None
        self.runner = runner
        self.sql = query
        self.filename = filename
        self._query = query
        self._cost = 0.0
        self._max_steps = config["database"]["sage_max_steps"]
        self._step = 0
        self._coverage = 0.0
        self._solutions = list()
        self._rewards = dict()
        self._tracker = QueryTracker()
        self._last_join_order = None
        self._conv_threshold = config["database"]["sage_convergence_threshold"]
        self._conv_step = 0

    @property
    def name(self) -> str:
        return self._query[0]

    @property
    def value(self) -> str:
        return self._query[1]

    @property
    def step(self) -> int:
        return self._step

    @property
    def coverage(self) -> float:
        return self._coverage

    @property
    def solutions(self) -> List[Dict[str, str]]:
        return self._solutions

    @solutions.setter
    def solutions(self, solutions: List[Dict[str, str]]) -> None:
        self._solutions = solutions

    @property
    def cost(self) -> float:
        return self._cost

    @cost.setter
    def cost(self, cost: float) -> None:
        self._cost = cost

    @property
    def rewards(self) -> Dict[str, Any]:
        return self._rewards

    @property
    def progression(self) -> float:
        if self._max_steps > 0:
            progress = max(self._step / self._max_steps, self._coverage)
        else:
            progress = self._coverage
        return round(progress * 100, 4)

    @property
    def complete(self) -> bool:
        if (self._max_steps > 0) and (self._step == self._max_steps):
            return True
        return self._coverage == 1.0

    @property
    def converged(self) -> bool:
        if self._conv_threshold == 0:
            return False
        return self._conv_step >= self._conv_threshold

    @property
    def tracker(self) -> QueryTracker:
        return self._tracker

    def report_progression(self, coverage: float) -> None:
        self._coverage = max(self._coverage, coverage)
        self._step += 1

    def report_solutions(self, solutions: List[Dict[str, str]]) -> None:
        self._solutions.extend(solutions)

    def reduce_solutions(self) -> List[Dict[str, str]]:
        def stringify_solution_mappings(solution_mappings: Dict[str, str]) -> str:
            mappings = ''
            variables = sorted(solution_mappings.keys())
            for variable in variables:
                mappings += f'{variable}:{solution_mappings[variable]}'
            return mappings

        memory = dict()
        unique_solutions = list()
        for solution_mappings in self._solutions:
            key = stringify_solution_mappings(solution_mappings)
            if key not in memory:
                unique_solutions.append(solution_mappings)
                memory[key] = None
        return unique_solutions

    def report_join_order(self, join_order: List[int]) -> None:
        if self._last_join_order == join_order:
            self._conv_step += 1
        else:
            self._last_join_order = join_order
            self._conv_step = 1

    def reset(self) -> None:
        self._step = 0
        self._coverage = 0.0
        self._solutions = list()
        self._rewards = dict()
        self._tracker = QueryTracker()
        self._last_join_order = None
        self._conv_step = 0

    def getDPlatency(self, forceLatency=False):
        if self.DPLantency == None:
            self.DPLantency = self.runner.getLatency(self,self.sql, forceLatency=forceLatency)
        return self.DPLantency
    def getDPPlantime(self,):
        if self.plTime == None:
            self.plTime = self.runner.getDPPlanTime(self,self.sql)
        return self.plTime
    def getDPCost(self,):
        if self.DPCost == None:
            self.DPCost = self.runner.getCost(self,self.sql)
        return self.DPCost
    def timeout(self,):
        if self.DPLantency == None:
            return 1000000
        return self.getDPlatency()*4+self.getDPPlantime()
    def getBestOrder(self,):
        return self.bestOrder
    def updateBestOrder(self,latency,order):
        if self.bestOrder == None or self.bestLatency > latency:
            self.bestLatency = latency
            self.bestOrder = order