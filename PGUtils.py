import psycopg2
import json
from math import log
class PGConfig:
    def __init__(self):
        self.keepExecutedPlan =True
        self.maxTimes = 5
        self.maxTime = 300000

LatencyDict = {}
selectivityDict = {}
LatencyRecordFileHandle = None

class PGRunner:
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

        self._dbname = dbname
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self.config = PGConfig()
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