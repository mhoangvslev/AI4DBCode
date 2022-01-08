import json
import os

from typing import Dict, Any

class Database():

    def __init__(self, config: Dict[str, Any]):
        relations = config['environment']['database']['relations']
        if os.path.exists(relations):
            with open(relations, 'r') as jsonfile:
                self._relations = json.load(jsonfile)
        else:
            message = f'The file {relations} is missing...'
            raise Exception(f'DatabaseError: {message}')

    def count_relations(self) -> int:
        return len(self._relations)

    def get_index(self, relation: str) -> int:
        if relation not in self._relations:
            message = f'The relation {relation} does not exists...'
            raise Exception(f'DatabaseError: {message}')
        else:
            return self._relations.index(relation)

    def get_relation(self, index: int) -> str:
        if index >= len(self._relations):
            message = f'No relation matches the index {index}...'
            raise Exception(f'DatabaseError: {message}')
        else:
            return self._relations[index]

    @staticmethod
    def extract_relation(
        triple_pattern: Dict[str, str], type: str = 'sp'
    ) -> str:
        subject = triple_pattern['subject']
        if triple_pattern['predicate'].startswith('?'):
            predicate = 'RDFTerm'
        else:
            predicate = triple_pattern['predicate']
        object = triple_pattern['object']
        if type == 'spo':
            return f'{subject} {predicate} {object}'
        elif type == 'sp':
            return f'{subject} {predicate} RDFTerm'
        elif type == 'p':
            return f'RDFTerm {predicate} RDFTerm'
        else:
            message = f'Cannot extract relations of type {type}'
            raise Exception(f'DatabaseError: {message}')
