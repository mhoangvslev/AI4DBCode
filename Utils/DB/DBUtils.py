import psycopg2
import json
import re
from math import log
from Utils.DB.Client import Client, ISQLTimeoutException, SaGeClient, VirtuosoClient

class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None

class DBRunner:
    def __init__(self, isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json") -> None:
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
        cost = self.getCost(sql,sqlwithplan, force_order=False)
        plTime = time.time()-startTime
        return plTime

    def getLatency(self, sql, sqlwithplan, force_order=False, forceLatency=False):
        raise NotImplementedError()
    
    def getCost(self, sql, sqlwithplan: str, force_order=False):
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
        
    def getLatency(self, sql, sqlwithplan: str, force_order=False, forceLatency=False):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """
        if self.isCostTraining and not forceLatency:
            return self.getCost(sql,sqlwithplan,force_order=force_order)
        global LatencyDict
        if self.isLatencyRecord and not forceLatency:
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

        if force_order:
            cursor.execute("set join_collapse_limit = 1;")

        cursor.execute("SET statement_timeout = "+str(int(sql.timeout()))+ ";")
        cursor.execute("set max_parallel_workers=1;")
        cursor.execute("set max_parallel_workers_per_gather = 1;")
        cursor.execute("set geqo_threshold = 20;")
        #cursor.execute("EXPLAIN "+sqlwithplan)
        thisQueryCost = self.getCost(sql,sqlwithplan, force_order=True)
        if thisQueryCost / sql.getDPCost()<100:
            try:
                cursor.execute("EXPLAIN ANALYZE "+sqlwithplan)
                rows = cursor.fetchall()
                row = rows[0][0]
                afterCost = float(rows[0][0].split("actual time=")[1].split("..")[1].split(" ")[0])
            except:
                conn.commit()
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlatency(),sql.timeout())
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlatency(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost

    def getCost(self,sql,sqlwithplan, force_order=False):
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

        if force_order:
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
        #     logging.debug(totalQuery)

        cursor.execute("EXPLAIN "+totalQuery)
        rows = cursor.fetchall()[0][0]
        #     logging.debug(rows)
        #     logging.debug(rows)
        total_rows = int(rows.split("rows=")[-1].split(" ")[0])

        resQuery = "select * from "+table+" Where "+whereCondition+";"
        # logging.debug(resQuery)
        cursor.execute("EXPLAIN  "+resQuery)
        rows = cursor.fetchall()[0][0]
        #     logging.debug(rows)
        select_rows = int(rows.split("rows=")[-1].split(" ")[0])
        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        #     logging.debug(stored_selectivity_fake[whereCondition],select_rows,total_rows)
        return selectivityDict[whereCondition]

class ISQLRunner(DBRunner):
    def __init__(self, endpoint, graph, host="localhost", port="1111", client="virtuoso", isCostTraining = True,latencyRecord = True,latencyRecordFile = "RecordFile.json"):
        super().__init__(isCostTraining=isCostTraining, latencyRecord=latencyRecord, latencyRecordFile=latencyRecordFile)
        
        if client not in ["virtuoso", "sage"]:
            raise ValueError("Configuration error: client must be either of ['virtuoso'; 'sage']!")

        self.dbClient: Client = (
            VirtuosoClient(endpoint=f"http://{host}:{port}/{endpoint}", graph=graph)
            if client == "virtuoso" else
            SaGeClient(endpoint=f"http://{host}:{port}/{endpoint}", graph=graph)
        )
        

    def getLatency(self, sql, sqlwithplan, force_order=False, forceLatency=False):
        """
        :param sql:a sqlSample object
        :return: the latency of sql
        """

        if self.isCostTraining and not forceLatency:
            return self.getCost(sql,sqlwithplan, force_order=force_order)
        global LatencyDict
        if self.isLatencyRecord and not forceLatency:
            if sqlwithplan in LatencyDict:
                return LatencyDict[sqlwithplan]
        
        thisQueryCost = self.getCost(sql,sqlwithplan, force_order=True)
        if thisQueryCost / sql.getDPCost()<100:
            try:
                afterCost = self.dbClient.query_latency(sqlwithplan, timeout=sql.timeout(), force_order=force_order)
            except ISQLTimeoutException:
                afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlatency(),sql.timeout())
        else:
            afterCost = max(thisQueryCost / sql.getDPCost()*sql.getDPlatency(),sql.timeout())
        afterCost += 5
        if self.isLatencyRecord:
            LatencyDict[sqlwithplan] =  afterCost
            global LatencyRecordFileHandle
            LatencyRecordFileHandle.write(json.dumps([sqlwithplan,afterCost])+"\n")
            LatencyRecordFileHandle.flush()
        return afterCost

    def getCost(self, sql, sqlwithplan, force_order=False):
        return self.dbClient.query_cost(sqlwithplan, force_order=force_order)
    
    def getSelectivity(self, table, whereCondition):
        global selectivityDict
        if whereCondition in selectivityDict:
            return selectivityDict[whereCondition]

        response = None

        try:
            totalQuery = f'SELECT * WHERE {{ {table} }}'
            response = self.dbClient.explain(totalQuery, mode=-1)
            total_rows = float(re.search(r'RDF_QUAD\w*\s+([0-9e\+\.]+)\srows', response).group(1))
        except:
            raise ValueError(f"Cannot execute query {totalQuery}. Response {response}")

        try:
            resQuery = f'SELECT * WHERE {{ {table} {whereCondition} }}'
            response = self.dbClient.explain(resQuery, mode=-1)
            select_rows = float(re.search(r'RDF_QUAD\w*\s+([0-9e\+\.]+)\srows', response).group(1))
        except:
            raise ValueError(f"Cannot execute query {resQuery}. Response: {response}")

        selectivityDict[whereCondition] = -log(select_rows/total_rows)
        return selectivityDict[whereCondition]