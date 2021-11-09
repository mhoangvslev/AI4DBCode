import os
import psycopg2
import json
import tempfile
import subprocess
import re
import logging
import time

from math import log
from SPARQLWrapper import SPARQLWrapper, JSON


class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None

class DBRunner(object):
    def __init__(self, isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json") -> None:
        super().__init__()
        self.isLatencyRecord = latencyRecord
        # self.LatencyRecordFileHandle = None
        global LatencyRecordFileHandle
        self.isCostTraining = isCostTraining
        if latencyRecord:
            LatencyRecordFileHandle = self.generateLatencyPool(latencyRecordFile)
    
    def generateLatencyPool(self,fileName):
        """
        :param fileName:
        :return:
        """
        import os
        import json
        if os.path.exists(fileName):
            f = open(fileName,"r")
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                global LatencyDict
                LatencyDict[data[0]] = data[1]
            f = open(fileName,"a")
        else:
            f = open(fileName,"w")
        return f

    def getDPPlanTime(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the planTime of sql
        """
        import time
        startTime = time.time()
        cost = self.getCost(sql,sqlwithplan)
        plTime = time.time()-startTime
        return plTime

    def getLatency(self, sql,sqlwithplan):
        raise NotImplementedError()
    
    def getCost(self,sql,sqlwithplan):
        raise NotImplementedError()    

    def getSelectivity(self,table,whereCondition):
        raise NotImplementedError()

class PGRunner(DBRunner):
    def __init__(self,dbname = '',user = '',password = '',host = '',port = '',isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        """
        :param dbname:
        :param user:
        :param password:
        :param host:
        :param port:
        :param latencyRecord:-1:loadFromFile
        :param latencyRecordFile:
        """

        super().__init__(isCostTraining=isCostTraining, latencyRecord=latencyRecord, latencyRecordFile=latencyRecordFile)

        self._dbname = dbname
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self.config = PGConfig()
        
    def getLatency(self, sql,sqlwithplan):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        # query = sql.toSql()
        if self.isCostTraining:
            return self.getCost(sql,sqlwithplan)
        global LatencyDict
        if self.isLatencyRecord:
            if sqlwithplan in LatencyDict:
                return LatencyDict[sqlwithplan]
        conn = psycopg2.connect(
            database=self._dbname, 
            user=self._user, 
            password=self._password, 
            host=self._host, 
            port=self._port
        )
        cursor = conn.cursor()

        cursor.execute("set join_collapse_limit = 1;")
        cursor.execute("SET statement_timeout = "+str(int(sql.timeout()))+ ";")
        cursor.execute("set max_parallel_workers=1;")
        cursor.execute("set max_parallel_workers_per_gather = 1;")
        cursor.execute("set geqo_threshold = 20;")
        cursor.execute("EXPLAIN "+sqlwithplan)
        thisQueryCost = self.getCost(sql,sqlwithplan)
        if thisQueryCost / sql.getDPCost()<100:
            try:
                cursor.execute("EXPLAIN ANALYZE "+sqlwithplan)
                rows = cursor.fetchall()
                row = rows[0][0]
                afterCost = float(rows[0][0].split("actual time=")[1].split("..")[1].split(" ")[
                                      0])
            except:
                conn.commit()
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlantecy(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost
    def getCost(self,sql,sqlwithplan):
        """
        :param sql: a sqlSample object
        :return: the cost of sql
        """

        conn = psycopg2.connect(
            database=self._dbname, 
            user=self._user, 
            password=self._password, 
            host=self._host, 
            port=self._port
        )
        cursor = conn.cursor()

        cursor.execute("set join_collapse_limit = 1;")
        cursor.execute("set max_parallel_workers=1;")
        cursor.execute("set max_parallel_workers_per_gather = 1;")
        cursor.execute("set geqo_threshold = 20;")
        cursor.execute("SET statement_timeout =  100000;")

        cursor.execute("EXPLAIN "+sqlwithplan)
        rows = cursor.fetchall()
        row = rows[0][0]
        afterCost = float(rows[0][0].split("cost=")[1].split("..")[1].split(" ")[
                              0])
        conn.commit()
        return afterCost

    def getSelectivity(self,table,whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]

        conn = psycopg2.connect(
            database=self._dbname, 
            user=self._user, 
            password=self._password, 
            host=self._host, 
            port=self._port
        )
        cursor = conn.cursor()
        
        cursor.execute("SET statement_timeout = "+str(int(100000))+ ";")
        totalQuery = "select * from "+table+";"
        #     print(totalQuery)

        cursor.execute("EXPLAIN "+totalQuery)
        rows = cursor.fetchall()[0][0]
        #     print(rows)
        #     print(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        # print(resQuery)
        cursor.execute("EXPLAIN  "+resQuery)
        rows = cursor.fetchall()[0][0]
        #     print(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        #     print(stored_selectivity_fake[whereCondition],select_rows,total_rows)
        return selectivityDict[whereCondition]

class ISQLWrapperException(Exception):
    pass

class ISQLWrapper(object):

    def __init__(self, hostname: str, username: str, password: str):
        self.hostname = hostname
        self.username = username
        self.password = password

    def execute_script(self, script: str):
        cmd = [f'{os.environ["VIRTUOSO_HOME"]}/bin/isql', self.hostname, self.username, self.password, script]
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

class ISQLRunner(DBRunner):
    def __init__(self, endpoint, graph, host="localhost", port="1111", isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        super().__init__(isCostTraining=isCostTraining, latencyRecord=latencyRecord, latencyRecordFile=latencyRecordFile)
        self._endpoint = f"http://{host}:{port}/{endpoint}"
        self._graph = graph
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
        try:
            response = sparql.query()
            solutions = response.convert()
            if 'x-exec-milliseconds' in response.info():
                solutions['complete'] = False
            else:
                solutions['complete'] = True
            return solutions
        except Exception as error:
            #logging.error(error)
            print(error)

    def _explain(self, query: str, force_order: bool = False):
        if force_order:
            query = self.__insert_force_order_pragma__(query)
        query = self.__insert_from_named_graph_clause__(query, self._graph)
        query = self.__remove_comments__(query)
        query = self.__format_regex__(query)
        cmd = f"select explain('sparql {query}', -7);"
        return self._isql.execute_cmd(cmd)

    
    def _query_cost(self, query: str, force_order: bool = False) -> float:
        response = self._explain(query, force_order=force_order)
        return float(response.split('\n')[9])
            
    def getLatency(self, sql, sqlwithplan):
        start_time = time.time()
        self._execute_query(sqlwithplan, force_order=True)
        end_time = time.time()
        return end_time - start_time

    def getCost(self, sql, sqlwithplan):
        return self._query_cost(sqlwithplan, force_order=True)
    
    def getSelectivity(self, table, whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]

        totalQuery = f'select * WHERE {{ ?s {table} ?p }};'
        total_rows = int(re.search(r'RDF_QUAD_POGS\s+([0-9]+)\srows', self._explain(totalQuery)).group(1))

        resQuery = f'select * WHERE {{ ?s {table} ?{whereCondition} }};'
        select_rows = int(re.search(r'RDF_QUAD_POGS\s+([0-9]+)\srows', self._explain(resQuery)).group(1))

        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        return selectivityDict[whereCondition]