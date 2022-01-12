from abc import ABC, abstractmethod
from SPARQLWrapper import SPARQLWrapper, JSON
import shutil
import requests
import subprocess
import tempfile
import re
import time
from typing import Dict, Any, List, Optional
import json
from tqdm import tqdm

from Utils.DB.Query import Query

class Client(ABC):

    def __init__(self, endpoint: str, graph: str):
        self._endpoint = endpoint
        self._graph = graph

    @abstractmethod
    def query_cost(self, query: str, force_order: bool = False) -> float:
        pass

    @abstractmethod
    def query_latency(self, query, timeout = 0, force_order=True) -> float:
        pass

    @abstractmethod
    def query_explain(self, query: str, force_order: bool = False, mode=-7) -> str:
        pass

    @abstractmethod
    def query_cardinality(self, query: str) -> float:
        pass


class ISQLWrapperException(Exception):
    pass

class ISQLWrapper(object):

    def __init__(self, hostname: str, username: str, password: str):
        self.hostname = hostname
        self.username = username
        self.password = password

    def execute_script(self, script: str):
        isql = shutil.which('isql')
        if isql is None: 
            result_url: str = requests.post(
                'http://127.0.0.1:4000/commands/isql', 
                json={"args": [self.hostname, self.username, self.password, script]}
            ).json()['result_url']

            result = requests.get(result_url.replace('wait=false', 'wait=true')).json()
            if result.get('report') is None:
                raise NotImplementedError(f'Testing for ISQL: {result.get("error")}')
            else:
                return result.get("report")
            
        else:  
            cmd = (isql + f' {self.hostname} {self.username} {self.password} {script}').split(' ')
            
            process = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if process.stderr:
                raise ISQLWrapperException(process.stderr)
            return process.stdout.decode('utf-8')

    def execute_cmd(self, cmd: str):
        if not cmd.endswith(';'):
            cmd += ';'
                
        tf_query = tempfile.NamedTemporaryFile()
        tf_query.write(cmd.encode('utf-8'))
        tf_query.flush()
        result = self.execute_script(tf_query.name)
        tf_query.close()
        return result

    def sparql_query(self, query: str):
        if not query.endswith(';'):
            query += ';'
        return self.execute_cmd("SPARQL " + query)

class ISQLTimeoutException(Exception):
    pass

class VirtuosoClient(Client):

    def __init__(self, endpoint: str, graph: str):
        super().__init__(endpoint, graph)
        host = next(filter(bool, re.search(r'((?:\d+\.){3}\d+)|(localhost)', endpoint).groups()))
        if host is None:
            raise ValueError(f"Invalid host {endpoint}")
        self._isql = ISQLWrapper(f"{host}:1111", "dba", "dba")

    def __insert_force_order_pragma__(self, query: str) -> str:
        return f'DEFINE sql:select-option "order" {query}'

    def __insert_from_named_graph_clause__(
        self, query: str, graph: str
    ) -> str:
        select_clause, where_clause = query.split('WHERE')
        return f'{select_clause} FROM <{graph}> WHERE {where_clause}'

    def __remove_comments__(self, query: str) -> str:
        return re.sub('\n[\t ]*#.*', '', query)

    def __format_regex__(self, query: str) -> str:
        query = re.sub('\'', '"', query)
        return re.sub(r'\\\\', r'\\', query)

    def _execute_query(
        self, query: str, timeout: int = 0, force_order: bool = False
    ) -> dict:
        if force_order:
            query = self.__insert_force_order_pragma__(query)
        # query = self.__remove_comments__(query)
        # query = self.__format_regex__(query)
        sparql = SPARQLWrapper(self._endpoint)
        sparql.setQuery(query)
        sparql.addDefaultGraph(self._graph)
        sparql.addParameter('timeout', f'{timeout}')
        sparql.setReturnFormat(JSON)
        
        start_time = time.time()
        response = sparql.query()
        elapsed_time = time.time() - start_time
        solutions = response.convert()
        if timeout > 0 and (elapsed_time * 1000) > timeout:
            raise ISQLTimeoutException
        if 'x-exec-milliseconds' in response.info():
            raise ISQLTimeoutException
        return solutions, elapsed_time

    def query_explain(self, query: str, force_order: bool = False, mode=-7) -> str:
        """[summary]

        Args:
            query (str): [description]
            force_order (bool, optional): [description]. Defaults to False.
            mode (int, optional): http://docs.openlinksw.com/virtuoso/fn_explain/. Defaults to -7.

        Returns:
            [type]: [description]
        """
        if force_order:
            query = self.__insert_force_order_pragma__(query)
        query = self.__insert_from_named_graph_clause__(query, self._graph)
        query = self.__remove_comments__(query)
        query = self.__format_regex__(query)
        cmd = f"select explain('sparql {query}', {mode});"
        return self._isql.execute_cmd(cmd)
    
    def query_cost(self, query: str, force_order: bool = False) -> float:
        response = self.query_explain(query, force_order=force_order)
        return float(response.split('\n')[9])

    def query_latency(self, query, timeout=0, force_order=True) -> float:
        _, afterCost = self._execute_query(query, timeout=timeout, force_order=force_order)
        return afterCost

    def query_cardinality(self, query: str) -> float:
        response = self.query_explain(query, mode=-1)
        return float(re.findall(r'RDF_QUAD\w*\s+([0-9e\+\.]+)\srows', response)[-1])
        #return float(re.search(r'RDF_QUAD\w*\s+([0-9e\+\.]+)\srows', response).group(1))

class SaGeClient(Client):

    def __init__(self, endpoint: str, graph: str):
        super().__init__(endpoint, graph)        

    def execute_query(
        self, query: str, next: str = None, quanta: int = 1,
        force_order: bool = False
    ) -> Dict[str, Any]:
        headers = {
            'accept': 'text/html',
            'content-type': 'application/json'
        }
        payload = {
            'query': query,
            'defaultGraph': self._graph,
            'next': next,
            'forceOrder': force_order
        }

        for _ in range(quanta):
            response = requests.post(
                self._endpoint, headers=headers, data=json.dumps(payload)
            ).json()
            if response['next'] is None:
                break
            payload['next'] = response['next']
        return response

    def query_explain(self, query: str, force_order: bool = False, mode=-7) -> float:
        payload = {
            'query': query,
            'defaultGraph': self._graph,
            'forceOrder': force_order
        }

        host = f'{self._endpoint}/explain'
        response = requests.post(host, data=json.dumps(payload)).json()
        return response['query']

    def query_cost(
        self, query: str, force_order: bool = False
    ) -> float:
        payload = {
            'query': query,
            'defaultGraph': self._graph,
            'next': None,
            'forceOrder': force_order
        }
        host = f'{self._endpoint}/explain'
        response = requests.post(host, data=json.dumps(payload)).json()
        return response['cost']

    def query_latency(self, query: str, timeout=0, force_order=True) -> float:

        def update_progressbar(
            progressbar: Optional[tqdm], query: Query, status: str
        ) -> None:
            if progressbar is None:
                return None
            progressbar.n = query.progression
            progressbar.set_postfix_str(
                f'query: {query.name}, coverage: {query.coverage}, state: {status}'
            )
            progressbar.refresh()

        query = Query(query=query)

        next = None
        progressbar = tqdm(total=100)

        while not query.complete:
            start = time.time()
            response = self.execute_query(
                query.value, next=next, force_order=force_order
            )
            elapsed_time = time.time() - start
            progression = response['stats']['metrics']['progression']
            solutions = response['bindings']
            next = response['next']
            query.report_solutions(solutions)
            query.report_progression(progression)
            update_progressbar(progressbar, query, 'running')

        progressbar.close()

        return elapsed_time

    def query_cardinality(self, query: str) -> float:
        payload = {
            'query': query,
            'defaultGraph': self._graph,
            'next': None,
            'forceOrder': False
        }

        host = f'{self._endpoint}/explain'
        response = requests.post(host, data=json.dumps(payload)).json()
        return response['cardinality']
