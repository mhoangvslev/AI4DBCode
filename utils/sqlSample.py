
from typing import List
from ImportantConfig import Config
from utils.DBUtils import DBRunner, PGRunner
from utils.JOBParser import DB, ComparisonISQL, ComparisonISQLEqual, ComparisonSQL, DummyTableISQL, FromTableISQL, FromTableSQL, TargetTableISQL, TargetTableSQL
from utils.TreeLSTM import SPINN
from utils.parser.parsed_query import ParsedQuery
from utils.parser.parser import QueryParser

import torch
import torch
import torch.nn as nn
from itertools import count
import numpy as np
from psqlparse import parse_dict
import os

config = Config()

class sqlInfo:
    def __init__(self, runner ,sql ,filename):
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

tree_lstm_memory = {}
class JoinTree:
    """Where the magic happens
    """
    def __init__(self, sqlt: sqlInfo, db_info: DB, runner: DBRunner, device):
        global tree_lstm_memory
        tree_lstm_memory  ={}
        self.sqlt = sqlt
        self.sql = self.sqlt.sql

        self.aliasname2fullname = {}
        self.runner = runner
        self.device = device
        self.aliasname2fromtable={}

        print(self.sqlt.filename, self.sql)

        if isinstance(runner, PGRunner):
            parse_result = parse_dict(self.sql)[0]["SelectStmt"]
            self.target_table_list = [TargetTableSQL(x["ResTarget"]) for x in parse_result["targetList"]]
            self.from_table_list = [FromTableSQL(x["RangeVar"]) for x in parse_result["fromClause"]]
       
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
                all_joins = {}

                subjects = []
                predicates = []
                objects = []
                for tp in tps:
                    s, p, o = tp['subject'], tp['predicate'], tp['object']
                    subjects.append(s)
                    predicates.append(p)
                    objects.append(o)
                
                for i, (s_i, p_i, o_i) in enumerate(zip(subjects, predicates, objects)):
                    other_tables.append(DummyTableISQL(s_i, p_i, o_i))
                    for j, (s_j, p_j, o_j) in enumerate(zip(subjects, predicates, objects)):
                        if i == j: continue
                        if s_i == o_j:
                            from_tables.append(FromTableISQL(s_j, p_j, o_j))
                            target_tables.append(TargetTableISQL(s_i, p_i, o_i))
                            all_joins[f"{p_j} |><| {p_i} ({ o_j })"] = {
                                "fromTable": p_j,
                                "targetTable": p_i,
                                "joinCol": o_j,
                                "fromOtherCol": s_j,
                                "targetOtherCol": o_i
                            }
                        
                        if o_i == s_j:
                            from_tables.append(FromTableISQL(s_i, p_i, o_i))
                            target_tables.append(TargetTableISQL(s_j, p_j, o_j))
                            all_joins[f"{p_i} |><| {p_j} ({ o_i })"] = {
                                "fromTable": p_i,
                                "targetTable": p_j,
                                "joinCol": o_i,
                                "fromOtherCol": s_i,
                                "targetOtherCol": o_j
                            }
                    
                return list(set(from_tables)), list(set(target_tables)), list(set(other_tables)), all_joins


            parse_result: ParsedQuery = QueryParser.parse(self.sql)
            #print(parse_result.triple_patterns)
            #print(parse_result.filters)
            self.from_table_list, self.target_table_list, self.other_table_list, self.all_joins = get_joins(parse_result.triple_patterns)

            all_join_tables = list(set(self.from_table_list + self.target_table_list + self.other_table_list))
            
            for table in all_join_tables:
                self.aliasname2fromtable[table.getAliasName()] = table
                self.aliasname2fullname[table.getAliasName()] = table.getFullName()
                #print(db_info.name2table.keys())
                db_info.name2table[table.getFullName()].updateTable(table)

            self.aliasnames = set(self.aliasname2fromtable.keys())
            self.comparison_list = [ComparisonISQL(x) for x in parse_result.filters]
            for join_name, join in self.all_joins.items():
                eq_comp = ComparisonISQLEqual(join_name, join).breakdown()
                self.comparison_list.extend(eq_comp)
            self.aliasnames_root_set = set([x.getAliasName() for x in all_join_tables])


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

        # Extract all Join and filters
        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = [0.0]*config.max_column_in_table*2

        self.join_candidate = set()
        self.join_matrix=[]
        for aliasname in self.aliasnames_root_set:
            self.join_list[aliasname] = []
        for idx in range(len(self.db_info)):
            self.join_matrix.append([0]*len(self.db_info))
            
        for comparison in self.comparison_list:
            """In SQL:
            - For each comparison t1.c1 = t2.c2 do:

            """

            # When the aliases are present on in to operands, 
            if len(comparison.aliasname_list) == 2:

                left_aliasname = comparison.aliasname_list[0]
                right_aliasname = comparison.aliasname_list[1]

                if os.environ['RTOS_ENGINE'] == "sparql":
                    tmp = list(filter(lambda x: left_aliasname in x, self.aliasname2fullname.keys()))
                    if len(tmp) > 1: 
                        raise ValueError(
                            f"The left aliasname {left_aliasname} should be unique! \n" +
                            f"Search candidates: {tmp} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    elif len(tmp) == 0: 
                        raise ValueError(
                            f"The left aliasname {left_aliasname} does not exist! \n" +
                            f"Search candidates: {self.aliasname2fullname.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    left_aliasname = tmp[0]

                if os.environ['RTOS_ENGINE'] == "sparql":
                    tmp = list(filter(lambda x: right_aliasname in x, self.aliasname2fullname.keys()))
                    if len(tmp) > 1: 
                        raise ValueError(
                            f"The right aliasname {right_aliasname} should be unique! \n" +
                            f"Search candidates: {tmp} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    elif len(tmp) == 0: 
                        raise ValueError(
                            f"The right aliasname {right_aliasname} does not exist! \n" +
                            f"Search candidates: {self.aliasname2fullname.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )                    
                    right_aliasname = tmp[0]

                if not left_aliasname in self.join_list:
                    self.join_list[left_aliasname] = []
                if not right_aliasname in self.join_list:
                    self.join_list[right_aliasname] = []

                # Create a join for left operand
                self.join_list[left_aliasname].append((right_aliasname,comparison))
                
                # Encode left operand
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]

                left_column = comparison.column_list[0]
                # if os.environ['RTOS_ENGINE'] == "sparql":
                #     tmp = list(filter(lambda x: left_column in x, left_table_class.column2idx.keys()))
                #     if len(tmp) > 1: 
                #         raise ValueError(
                #             f"The left column {left_column} should be unique! \n" +
                #             f"Search candidates: {left_table_class.column2idx.keys()} \n"
                #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                #         )
                #     elif len(tmp) == 0: 
                #         raise ValueError(
                #             f"The left column {left_column} does not exist! \n" +
                #             f"Search candidates: {left_table_class.column2idx.keys()} \n"
                #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                #         )
                #     left_column = tmp[0]

                table_idx = left_table_class.column2idx[left_column]

                self.table_fea_set[left_aliasname][table_idx * 2] = 1
                self.join_list[right_aliasname].append((left_aliasname,comparison))

                # Encode right operand
                right_fullname = self.aliasname2fullname[right_aliasname]
                right_table_class = db_info.name2table[right_fullname]

                right_column = comparison.column_list[1]
                
                # if os.environ['RTOS_ENGINE'] == "sparql":
                #     tmp = list(filter(lambda x: right_column in x, right_table_class.column2idx.keys()))
                #     if len(tmp) > 1: 
                #         raise ValueError(
                #             f"The right column {right_column} should be unique! \n" +
                #             f"Search candidates: {tmp} \n"
                #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                #         )
                #     elif len(tmp) == 0: 
                #         raise ValueError(
                #             f"The right column {right_column} does not exist! \n" +
                #             f"Search candidates: {right_table_class.column2idx.keys()} \n"
                #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                #         )
                #     right_column = tmp[0]

                table_idx = right_table_class.column2idx[right_column]
                self.table_fea_set[right_aliasname][table_idx * 2] = 1

                # Add two join candidate left-right then right-left. While they yield the same result but one is faster than the another.
                self.join_candidate.add((left_aliasname,right_aliasname))
                self.join_candidate.add((right_aliasname,left_aliasname))
                
                # Update M matrix
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            else:
                try: 
                    left_aliasname = comparison.aliasname_list[0]
                except IndexError:
                    raise IndexError(f"Error while parsing {comparison}. Alias list: {comparison.aliasname_list}")
                    
                if os.environ['RTOS_ENGINE'] == "sparql":
                    tmp = list(filter(lambda x: left_aliasname in x, self.aliasname2fullname.keys()))
                    if len(tmp) > 1: 
                        raise ValueError(
                            f"The left aliasname {left_aliasname} should be unique! \n" +
                            f"Search candidates: {tmp} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    elif len(tmp) == 0: 
                        raise ValueError(
                            f"The left aliasname {left_aliasname} does not exist! \n" +
                            f"Search candidates: {self.aliasname2fullname.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    left_aliasname = tmp[0]

                if not left_aliasname in self.filter_list:
                    self.filter_list[left_aliasname] = []
                self.filter_list[left_aliasname].append(comparison)
                left_fullname = self.aliasname2fullname[left_aliasname]
                left_table_class = db_info.name2table[left_fullname]

                left_column = comparison.column_list[0]
                if os.environ['RTOS_ENGINE'] == "sparql":
                    tmp = list(filter(lambda x: left_column == x, left_table_class.column2idx.keys()))
                    if len(tmp) > 1: 
                        raise ValueError(
                            f"The left column {left_column} should be unique! \n" +
                            f"Search candidates: {left_table_class.column2idx.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    if len(tmp) < 1: 
                        raise ValueError(
                            f"The left column {left_column} does not exist! \n" +
                            f"Search candidates: {left_table_class.column2idx.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    left_column = tmp[0]

                table_idx = left_table_class.column2idx[left_column]
                self.table_fea_set[left_aliasname][table_idx * 2 + 1] += self.runner.getSelectivity(
                    str(self.aliasname2fromtable[left_aliasname]), 
                    comparison.toString()
                )


        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = self.device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = set([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = set()
                self.aliasnames_join_set[aliasname].add(y[0])


        predice_list_dict={}
        for table in self.db_info.tables:
            predice_list_dict[table.name] = [0] * len(table.column2idx)
        for filter_table in self.filter_list:
            for comparison in self.filter_list[filter_table]:
                aliasname = comparison.aliasname_list[0]

                if os.environ['RTOS_ENGINE'] == "sparql":
                    tmp = list(filter(lambda x: aliasname in x, self.aliasname2fullname.keys()))
                    if len(tmp) > 1: 
                        raise ValueError(
                            f"The aliasname {aliasname} should be unique! \n" +
                            f"Search candidates: {self.aliasname2fullname.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    if len(tmp) < 1: 
                        raise ValueError(
                            f"The aliasname {aliasname} does not exist! \n" +
                            f"Search candidates: {self.aliasname2fullname.keys()} \n"
                            f"Comparison: {comparison}, type: {type(comparison)} \n"
                        )
                    aliasname = tmp[0]

                fullname = self.aliasname2fullname[aliasname]
                table = self.db_info.name2table[fullname]
                for column in comparison.column_list:

                    # if os.environ['RTOS_ENGINE'] == "sparql":
                    #     tmp = list(filter(lambda x: column in x, table.column2idx.keys()))
                    #     if len(tmp) > 1: 
                    #         raise ValueError(
                    #             f"The right column {right_column} should be unique! \n" +
                    #             f"Search candidates: {table.column2idx.keys()} \n"
                    #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                    #         )
                    #     if len(tmp) < 1: 
                    #         raise ValueError(
                    #             f"The right column {right_column} does not exist! \n" +
                    #             f"Search candidates: {table.column2idx.keys()} \n"
                    #             f"Comparison: {comparison}, type: {type(comparison)} \n"
                    #         )
                    #     column = tmp[0]

                    columnidx = table.column2idx[column]
                    predice_list_dict[self.aliasname2fullname[filter_table]][columnidx] = 1
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

    def recTableISQL(self,node) -> str:
        res = []
        if isinstance(node,int):

            left_son = self.left_son[node]
            right_son = self.right_son[node]

            leftRes = self.recTableISQL(left_son)
            if left_son not in self.aliasnames:
                #raise ValueError(f"Left child {left_son}({leftRes}) not in {self.aliasnames}")
                #leftRes = leftRes.splitlines()[-1]
                res.extend(leftRes.splitlines())
            else:
                res.append(leftRes)

            # Immediately follows by filters
            filter_list = []
            on_list = []
            if left_son in self.filter_list:
                for condition in self.filter_list[left_son]:
                    filter_list.append(condition.toString())

            if right_son in self.filter_list :
                for condition in self.filter_list[right_son]:
                    filter_list.append(condition.toString())
            
            # ... then comparisons
            cpList = []
            joined_aliasname = set([self.left_aliasname[node],self.right_aliasname[node]])
            for left_table in self.aliasnames_set[left_son]:
                for right_table, comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[right_son]:
                        if (comparison.aliasname_list[1] in joined_aliasname and comparison.aliasname_list[0] in joined_aliasname):
                            cpList.append(comparison.toString())
                        else:
                            on_list.append(comparison.toString())
            
            # inner join
            if len(filter_list + on_list + cpList) > 0:
                res.append(self.recTableISQL(right_son))
                res.extend(cpList + on_list + filter_list)
            
            # cross join
            else:
                res.append(self.recTableISQL(right_son))

            return "\n".join(res)

        elif isinstance(node, str):
            return node
        else:
            raise NotImplementedError(f'Cannot handle node of unknown type {type(node)}')

    def recTableSQL(self,node):
        if isinstance(node,int):
            res =  "("
            leftRes = self.recTableSQL(self.left_son[node])
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
                res += self.recTableSQL(self.right_son[node])
                res += "\non "
                res += " AND ".join(cpList + on_list+filter_list)
            else:
                res += "cross join "
                res += self.recTableSQL(self.right_son[node])

            res += ")"
            return res
        else:
            return str(self.aliasname2fromtable[node])
    
    def encode_tree_regular(self, model: SPINN, node_idx):

        def get_inputX(node: int):

            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]

            offset = int(config.n_words/2)

            left_node_id = self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]
            right_node_id = self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]

            left_emb = model.leaf(
                torch.tensor([left_node_id + offset], device = self.device),
                self.table_fea_set[left_aliasname]
            )
            
            right_emb = model.leaf(
                torch.tensor([right_node_id + offset], device = self.device),
                self.table_fea_set[right_aliasname]
            )
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
                node_id = [self.db_info.name2idx[self.aliasname2fullname[node]]]
                res = model.leaf(
                    torch.tensor(node_id, device = self.device),
                    self.table_fea_set[node]
                )
                tree_lstm_memory[node] = res

            return res
        encoding, _ = encode_node(node_idx)
        return encoding
    def encode_tree_fold(self,fold, node_idx):
        def get_inputX(node):
            left_aliasname = self.left_aliasname[node]
            right_aliasname = self.right_aliasname[node]

            offset = int(config.n_words/2)

            left_emb,c1 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[left_aliasname]]+offset,self.table_fea_set[left_aliasname]).split(2)
            right_emb,c2 = fold.add('leaf',self.db_info.name2idx[self.aliasname2fullname[right_aliasname]]+offset,self.table_fea_set[right_aliasname]).split(2)
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

    def toSql(self):
        root = self.total - 1
        print(f"Root: {root}")

        if os.environ["RTOS_ENGINE"] == "sql":
            res = "select "+",\n".join([str(x) for x in self.target_table_list])+"\n"
            res += "from " + self.recTableSQL(root)[1:-1]
            res += ";"
        else:
            res = "SELECT * WHERE { \n\t" 
            res += ' .\n\t'.join(set(self.recTableISQL(root).splitlines()))
            res += "\n};"

        print("Proposed plan:")
        print(res)
        return res

    def plan2Cost(self):
        sql = self.toSql()
        return self.runner.getLatency(self.sqlt, sql)





