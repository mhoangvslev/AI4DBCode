from typing import List, Dict, Set
from rdflib.plugins.sparql.parserutils import Expr

from utils.database import Database


class ParsedQuery():

    def __init__(
        self,
        projection: List[str],
        filters: List[Expr],
        values: List[Dict[str, str]],
        triple_patterns: List[Dict[str, str]],
        constrained_variables: List[str]
    ):
        self._projection = projection
        self._filters = filters
        self._values = values
        self._triple_patterns = triple_patterns
        self._contrained_variables = constrained_variables
        self._rel2tp_index = None
        self._tp2rel_index = None
        self._rel_index = None

    @property
    def projection(self) -> List[str]:
        return self._projection

    @property
    def filters(self) -> List[Expr]:
        return self._filters

    @property
    def values(self) -> List[Dict[str, str]]:
        return self._values

    @property
    def triple_patterns(self) -> List[Dict[str, str]]:
        return self._triple_patterns

    @property
    def constrained_variables(self) -> Set[str]:
        return self._contrained_variables

    @property
    def relation_indexes(self) -> List[int]:
        if self._rel_index is None:
            raise Exception('The rel2tp index has not been created')
        return self._rel_index

    def is_constrained_variables(self, variable: str) -> bool:
        return variable in self._contrained_variables

    def get_triple_pattern(self, relation_index: int) -> Dict[str, str]:
        if self._rel2tp_index is None:
            raise Exception('The rel2tp index has not been created')
        return self._rel2tp_index[relation_index]

    def get_relation_index(self, triple_pattern: Dict[str, str]) -> int:
        if self._tp2rel_index is None:
            raise Exception('The tp2rel index has not been created')
        return self._tp2rel_index[Database.extract_relation(triple_pattern)]

    def create_relation_indexes(self, database: Database) -> None:
        self._rel2tp_index = dict()
        self._tp2rel_index = dict()
        self._rel_index = list()
        for triple_pattern in self._triple_patterns:
            relation = Database.extract_relation(triple_pattern)
            relation_index = database.get_index(relation)
            self._rel2tp_index[relation_index] = triple_pattern
            self._tp2rel_index[relation] = relation_index
            self._rel_index.append(relation_index)
