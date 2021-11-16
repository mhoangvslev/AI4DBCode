
from utils.DBUtils import DBRunner, PGRunner
from utils.JOBParser import DB, ComparisionISQL, ComparisonSQL, FromTableISQL, TargetTable,FromTable,Comparison, TargetTableISQL
from utils.parser.parsed_query import ParsedQuery
from utils.parser.parser import QueryParser
max_column_in_table = 15
import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
from psqlparse import parse_dict
import os

tree_lstm_memory = {}
class JoinTree:
    """Where the magic happens
    """
    def __init__(self, sqlt, db_info: DB, runner: DBRunner, device):
        global tree_lstm_memory
        tree_lstm_memory  ={}
        self.sqlt = sqlt
        self.sql = self.sqlt.sql

        self.aliasname2fullname = {}
        self.runner = runner
        self.device = device
        self.aliasname2fromtable={}

        if isinstance(runner, PGRunner):
            parse_result = parse_dict(self.sql)[0]["SelectStmt"]
            self.target_table_list = [TargetTable(x["ResTarget"]) for x in parse_result["targetList"]]
            self.from_table_list = [FromTable(x["RangeVar"]) for x in parse_result["fromClause"]]
       
            for table in self.from_table_list:
                self.aliasname2fromtable[table.getAliasName()] = table
                self.aliasname2fullname[table.getAliasName()] = table.getFullName()
            self.aliasnames = set(self.aliasname2fromtable.keys())
            self.comparison_list =[ComparisonSQL(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
            self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])

        else:
            """
            # Note to dev: 
            1. The parsed query looks like this:
            [
                {
                    'subject': "s",
                    'predicate': "p",
                    'object': "o"
                },
                ...
            ]

            example:
            [
                {'subject': '?mi', 'predicate': 'http://imdb.org/movie_info#movie_id', 'object': '?t', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mi', 'predicate': 'http://imdb.org/movie_info#info_type_id', 'object': '?it1', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?midx', 'predicate': 'http://imdb.org/movie_info_idx#movie_id', 'object': '?t', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?midx', 'predicate': 'http://imdb.org/movie_info_idx#info_type_id', 'object': '?it2', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mc', 'predicate': 'http://imdb.org/movie_companies#movie_id', 'object': '?t', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mc', 'predicate': 'http://imdb.org/movie_companies#company_type_id', 'object': '?ct', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mc', 'predicate': 'http://imdb.org/movie_companies#company_id', 'object': '?c', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mc', 'predicate': 'http://imdb.org/movie_companies#movie_info#movie_id#movie_id', 'object': '?mi', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mc', 'predicate': 'http://imdb.org/movie_companies#movie_info_idx#movie_id#movie_id', 'object': '?midx', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mi', 'predicate': 'http://imdb.org/movie_info#movie_info_idx#movie_id#movie_id', 'object': '?midx', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?mi', 'predicate': 'http://imdb.org/movie_info#info', 'object': '?mi_info', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?midx', 'predicate': 'http://imdb.org/movie_info_idx#info', 'object': '?midx_info', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?c', 'predicate': 'http://imdb.org/company_name#name', 'object': '?c_name', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?t', 'predicate': 'http://imdb.org/title_t#title', 'object': '?t_title', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?t', 'predicate': 'http://imdb.org/title_t#production_year', 'object': '?t_production_year', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?c', 'predicate': 'http://imdb.org/company_name#country_code', 'object': '"[us]"', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?ct', 'predicate': 'http://imdb.org/company_type#kind', 'object': '"production companies"', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?it1', 'predicate': 'http://imdb.org/info_type#info', 'object': '"genres"', 'graph': 'http://localhost:8890/sparql'}, 
                {'subject': '?it2', 'predicate': 'http://imdb.org/info_type#info', 'object': '"rating"', 'graph': 'http://localhost:8890/sparql'}
            ]

            2. An alias is created by considering the entire triple pattern:
                alias = hash(spo string)
            
            3. Axioms:
               - A "table" in SQL corresponds to a "predicate" in SPARQL. 
               - A SQL "column" corresponds to either "subject" or "predicate" in SPARQL.
            """

            def get_joins(tps):

                from_tables = []
                target_tables = []
                other_tables = []

                subjects = []
                predicates = []
                objects = []
                for tp in tps:
                    subjects.append(tp['subject'])
                    predicates.append(tp['predicate'])
                    objects.append(tp['object'])
                
                for i, (s_i, p_i, o_i) in enumerate(zip(subjects, predicates, objects)):
                    for j, (s_j, p_j, o_j) in enumerate(zip(subjects, predicates, objects)):
                        if i == j: continue
                        if s_i == o_j:
                            from_tables.append(FromTableISQL(p_j, f'{s_j} <{p_j}> {o_j}'))
                            target_tables.append(TargetTableISQL(p_i, f'{s_i} <{p_i}> {o_i}'))
                        
                        elif o_i == s_j:
                            from_tables.append(FromTableISQL(p_i, f'{s_i} <{p_i}> {o_i}'))
                            target_tables.append(TargetTableISQL(p_j, f'{s_j} <{p_j}> {o_j}'))
                        else:
                            other_tables.append(FromTableISQL(p_i, f'{s_i} <{p_i}> {o_i}'))
                return from_tables, target_tables, other_tables


            parse_result: ParsedQuery = QueryParser.parse(self.sql)
            #print(parse_result.triple_patterns)
            #print(parse_result.filters)
            self.from_table_list, self.target_table_list, self.other_table_list = get_joins(parse_result.triple_patterns)
            
            for table in self.from_table_list + self.target_table_list + self.other_table_list:
                self.aliasname2fromtable[table.getAliasName()] = table
                self.aliasname2fullname[table.getAliasName()] = table.getFullName()
                db_info.tables[table.getFullName()].updateTable(table.getAliasName())
            self.aliasnames = set(self.aliasname2fromtable.keys())
            self.comparison_list = [ComparisionISQL(x) for x in parse_result.filters]
            self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list + self.target_table_list + self.other_table_list])


        self.db_info = db_info
        self.join_list = {}
        self.filter_list = {}

        self.aliasnames_fa = {}
        self.aliasnames_set = {}
        self.aliasnames_join_set = {}
        self.left_son = {}
        self.right_son = {}
        self.total = 0
        self.left_aliasname = {}
        self.right_aliasname = {}

        self.table_fea_set = {}
        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = [0.0]*max_column_in_table*2

        # Extract all Join and filters
        self.join_candidate = set()
        self.join_matrix=[]
        for aliasname in self.aliasnames_root_set:
            self.join_list[aliasname] = []
        for idx in range(len(self.db_info)):
            self.join_matrix.append([0]*len(self.db_info))
        for comparison in self.comparison_list:

            def get_comparision_info(mode):
                """[summary]

                Args:
                    mode ([type]): left = 0, right = 1

                Returns:
                    [type]: [description]
                """
                aliasname = comparison.aliasname_list[mode]
                aliasname = list(filter(lambda x: aliasname in x, self.aliasnames))[0]

                if os.environ["RTOS_ENGINE"] == "sparql":
                    fullname = dict(filter(lambda x: aliasname in x[0], self.aliasname2fullname.items()))
                    fullname = list(fullname.values())[0]
                else:
                    fullname = self.aliasname2fullname[aliasname]

                table_class = db_info.name2table[fullname]

                if os.environ["RTOS_ENGINE"] == "sparql":
                    #print(left_table_class.column2idx.keys(), comparison.column_list[0])
                    table_idx = dict(filter(lambda x: comparison.column_list[mode] in x[0], table_class.column2idx.items()))
                    table_idx = list(table_idx.values())[0]
                else:
                    table_idx = table_class.column2idx[comparison.column_list[mode]]
                return aliasname, fullname, table_idx

            if len(comparison.aliasname_list) == 2:
                if not comparison.aliasname_list[0] in self.join_list:
                    self.join_list[comparison.aliasname_list[0]] = []
                if not comparison.aliasname_list[1] in self.join_list:
                    self.join_list[comparison.aliasname_list[1]] = []
                self.join_list[comparison.aliasname_list[0]].append((comparison.aliasname_list[1],comparison))
                left_aliasname, left_fullname, left_table_idx = get_comparision_info(0)
                self.table_fea_set[left_aliasname][left_table_idx * 2] = 1

                self.join_list[comparison.aliasname_list[1]].append((comparison.aliasname_list[0],comparison))
                right_aliasname, right_fullname, right_table_idx = get_comparision_info(1)            
                self.table_fea_set[right_aliasname][right_table_idx * 2] = 1

                self.join_candidate.add((comparison.aliasname_list[0],comparison.aliasname_list[1]))
                self.join_candidate.add((comparison.aliasname_list[1],comparison.aliasname_list[0]))
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            
            #elif len(comparison.aliasname_list) == 1:
            else:
                if not comparison.aliasname_list[0] in self.filter_list:
                    self.filter_list[comparison.aliasname_list[0]] = []
                self.filter_list[comparison.aliasname_list[0]].append(comparison)

                left_aliasname, left_fullname, left_table_idx = get_comparision_info(0)
                comp_alias = dict(filter(lambda x: comparison.aliasname_list[0] in x[0], self.aliasname2fromtable.items()))
                comp_alias = list(comp_alias.keys())[0]

                #print(comp_alias)

                self.table_fea_set[left_aliasname][left_table_idx * 2 + 1] += self.runner.getSelectivity(
                    str(self.aliasname2fromtable[comp_alias]),
                    str(comparison)
                )


        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = self.device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])


        predice_list_dict={}
        for table in self.db_info.tables.values():
            predice_list_dict[table.name] = [0] * len(table.column2idx)
        for filter_table in self.filter_list:
            for comparison in self.filter_list[filter_table]:
                aliasname = comparison.aliasname_list[0]
                fullname = dict(filter(lambda x: aliasname in x[0], self.aliasname2fullname.items()))
                fullname = list(fullname.values())[0]
                
                table = dict(filter(lambda x: fullname in x[0], self.db_info.name2table.items()))
                table = list(table.values())[0]
                for column in comparison.column_list:
                    columnidx = dict(filter(lambda x: column in x[0], table.column2idx.items()))
                    columnidx = list(columnidx.values())[0]

                    filter_name = dict(filter(lambda x: filter_table in x[0], self.aliasname2fullname.items()))
                    filter_name = list(filter_name.keys())[0]
                    predice_list_dict[self.aliasname2fullname[filter_name]][columnidx] = 1

        self.predice_feature = []
        for fullname in predice_list_dict:
            self.predice_feature+= predice_list_dict[fullname]

        """
            self.join_matrix is the attribute matrix for column c in neural network (denoted M in the paper)
            self.table_fea_set is the feature vector of column c (denoted F(C) in the paper)
            R is representation for column (c), state (s), join tree (T) or query (q)
        """

        self.predice_feature = np.asarray(self.predice_feature).reshape(1,-1)
        self.join_matrix = torch.tensor(np.asarray(self.join_matrix).reshape(1,-1),device = self.device,dtype = torch.float32)

    def resetJoin(self):
        self.aliasnames_fa = {}
        self.left_son = {}
        self.right_son = {}
        self.aliasnames_root_set = set([x.getAliasName() for x in self.from_table_list])

        self.left_aliasname  = {}
        self.right_aliasname =  {}
        self.aliasnames_join_set = {}
        for aliasname in self.aliasnames_root_set:
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])

        self.total = 0
    def findFather(self,node_name):
        fa_name = node_name
        while  fa_name in self.aliasnames_fa:
            fa_name = self.aliasnames_fa[fa_name]
        while  node_name in self.aliasnames_fa:
            temp_name = self.aliasnames_fa[node_name]
            self.aliasnames_fa[node_name] = fa_name
            node_name = temp_name
        return fa_name

    def joinTables(self,aliasname_left,aliasname_right,fake=False):
        aliasname_left_fa = self.findFather(aliasname_left)
        aliasname_right_fa = self.findFather(aliasname_right)
        self.aliasnames_fa[aliasname_left_fa] = self.total
        self.aliasnames_fa[aliasname_right_fa] = self.total
        self.left_son[self.total] = aliasname_left_fa
        self.right_son[self.total] = aliasname_right_fa
        self.aliasnames_root_set.add(self.total)

        self.left_aliasname[self.total] = aliasname_left
        self.right_aliasname[self.total] = aliasname_right
        if not fake:
            self.aliasnames_set[self.total] = self.aliasnames_set[aliasname_left_fa]|self.aliasnames_set[aliasname_right_fa]
            self.aliasnames_join_set[self.total] = (self.aliasnames_join_set[aliasname_left_fa]|self.aliasnames_join_set[aliasname_right_fa])-self.aliasnames_set[self.total]
            self.aliasnames_root_set.remove(aliasname_left_fa)
            self.aliasnames_root_set.remove(aliasname_right_fa)

        self.total += 1
    def recTable(self,node):
        if isinstance(node,int):
            res =  "("
            leftRes = self.recTable(self.left_son[node])
            if not self.left_son[node] in self.aliasnames:
                leftRes = leftRes[1:-1]

            res += leftRes + "\n"
            filter_list = []
            on_list = []
            if self.left_son[node] in self.filter_list:
                for condition in self.filter_list[self.left_son[node]]:
                    filter_list.append(str(condition))

            if self.right_son[node] in self.filter_list :
                for condition in self.filter_list[self.right_son[node]]:
                    filter_list.append(str(condition))

            cpList = []
            joined_aliasname = set([self.left_aliasname[node],self.right_aliasname[node]])
            for left_table in self.aliasnames_set[self.left_son[node]]:
                for right_table,comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[self.right_son[node]]:
                        if (comparison.aliasname_list[1] in joined_aliasname and comparison.aliasname_list[0] in joined_aliasname):
                            cpList.append(str(comparison))
                        else:
                            on_list.append(str(comparison))
            if len(filter_list+on_list+cpList)>0:
                res += "inner join "
                res += self.recTable(self.right_son[node])
                res += "\non "
                res += " AND ".join(cpList + on_list+filter_list)
            else:
                res += "cross join "
                res += self.recTable(self.right_son[node])

            res += ")"
            return res
        else:
            return str(self.aliasname2fromtable[node])
    def encode_tree_regular(self,model, node_idx):

        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            left_emb =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25],device = self.device),self.table_fea_set[left_aliasname])
            right_emb = model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25],device = self.device),self.table_fea_set[right_aliasname])
            return model.inputX(left_emb[0],right_emb[0])
        def encode_node(node):
            if node in tree_lstm_memory:
                return tree_lstm_memory[node]
            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                res =  model.childrenNode(left_h, left_c, right_h, right_c,inputX)
                if self.total > node + 1:
                    tree_lstm_memory[node] = res
            else:
                res =  model.leaf(torch.tensor([self.db_info.name2idx[self.aliasname2fullname[node]]],device = self.device),self.table_fea_set[node])
                tree_lstm_memory[node] = res

            return res
        encoding, _ = encode_node(node_idx)
        return encoding
    def encode_tree_fold(self,fold, node_idx):
        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]
            left_emb,c1 =  fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+25,self.table_fea_set[left_aliasname]).split(2)
            right_emb,c2 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+25,self.table_fea_set[right_aliasname]).split(2)
            return fold.add('inputX',left_emb,right_emb)
        def encode_node(node):

            if isinstance(node,int):
                left_h, left_c = encode_node(self.left_son[node])
                right_h, right_c = encode_node(self.right_son[node])
                inputX = get_inputX(node)
                return fold.add('childrenNode',left_h, left_c, right_h, right_c,inputX).split(2)
            else:
                return fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[node]],self.table_fea_set[node]).split(2)
        encoding, _ = encode_node(node_idx)
        return encoding
    def toSql(self, format):
        root = self.total - 1
        res = "select " + ",\n".join([str(x) for x in self.target_table_list]) + "\n"

        if format == "sql":
            res += "FROM " + self.recTable(root)[1:-1]
        elif format == "sparql":
            res += f"WHERE { {self.recTable(root)[1:-1]} }"

        res += ";"

        return res

    def plan2Cost(self):
        sql = self.toSql(os.environ['RTOS_ENGINE'])
        return self.runner.getLatency(self.sqlt, sql)

class sqlInfo:
    def __init__(self,runner,sql,filename):
        self.DPLantency = None
        self.DPCost = None
        self.bestLatency = None
        self.bestCost = None
        self.bestOrder = None
        self.plTime = None
        self.runner = runner
        self.sql = sql
        self.filename = filename

    def getDPlatency(self,):
        if self.DPLantency == None:
            self.DPLantency = self.runner.getLatency(self,self.sql)
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





