import sage.query_engine.optimizer.utils as utils

from typing import Dict, Any
from rdflib.plugins.sparql.parserutils import CompValue, Expr

from sage.query_engine.optimizer.parser import Parser
from sage.query_engine.optimizer.logical.plan_visitor import (
    LogicalPlanVisitor, TriplePattern
)
from sage.query_engine.optimizer.logical.visitors.filter_splitter import (
    FilterSplitter
)
from sage.query_engine.optimizer.physical.visitors.filter_push_down import (
    FilterVariablesExtractor
)

from Utils.parser.parsed_query import ParsedQuery


class QueryParser(LogicalPlanVisitor):

    def __init__(self, query: str):
        self._projection = list()
        self._filters = list()
        self._values = list()
        self._triple_patterns = list()
        self._contrained_variables = set()
        self.visit(FilterSplitter().visit(Parser().parse(query)))

    @staticmethod
    def parse(query: str) -> ParsedQuery:
        query_features = QueryParser(query)
        return ParsedQuery(
            query_features._projection,
            query_features._filters,
            query_features._values,
            query_features._triple_patterns,
            query_features._contrained_variables
        )

    def visit_select_query(self, node: CompValue) -> Any:
        self.visit(node.p)

    def visit_projection(self, node: CompValue) -> Any:
        self._projection = list(map(lambda t: '?' + str(t), node.PV))
        self.visit(node.p)

    def visit_join(self, node: CompValue) -> Any:
        self.visit(node.p1)
        self.visit(node.p2)

    def visit_union(self, node: CompValue) -> Any:
        self.visit(node.p1)
        self.visit(node.p2)

    def visit_filter(self, node: CompValue) -> Any:
        node.expr.vars = FilterVariablesExtractor().visit(node.expr)
        self._filters.append(node.expr)
        self._contrained_variables.update(node.expr.vars)
        self.visit(node.p)

    def visit_to_multiset(self, node: CompValue) -> Any:
        self.visit(node.p)

    def visit_values(self, node: CompValue) -> Any:
        self._values.append(utils.format_solution_mappings(node.res))

    def visit_bgp(self, node: CompValue) -> Any:
        for triple_pattern in node.triples:
            self.visit(triple_pattern)

    def visit_scan(self, node: TriplePattern) -> Any:
        triple_pattern = {
            'subject': utils.format_term(node[0]),
            'predicate': utils.format_term(node[1]),
            'object': utils.format_term(node[2]),
            'graph': "http://localhost:8890/sparql"
        }
        self._triple_patterns.append(triple_pattern)
    
