import logging
from typing import Dict, List, Set, Tuple, Union
from collections_extended.setlists import SetList
from torch._C import device
from torchfold.torchfold import Fold

from ImportantConfig import Config
from utils.DBUtils import DBRunner, ISQLRunner, PGRunner
from utils.JOBParser import DB, ComparisonISQL, ComparisonISQLEqual, ComparisonSQL, DummyTableISQL, FromTableISQL, FromTableSQL, JoinISQL, TargetTableISQL, TargetTableSQL
from utils.TreeLSTM import SPINN
from utils.parser.parsed_query import ParsedQuery
from utils.parser.parser import QueryParser

import torch
import re
import torch.nn as nn
from itertools import count
import numpy as np
from psqlparse import parse_dict
import os
import graphviz as gv
from collections_extended import setlist

config = Config()

NB_FEATURE_SLOTS = 2 if os.environ["RTOS_ENGINE"] == "sql" else 4

class sqlInfo:
    def __init__(self, runner: DBRunner, sql: str, filename: str):
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
    def __init__(self, sqlt: sqlInfo, db_info: DB, runner: DBRunner, device: device):
        global tree_lstm_memory
        self.nbFilters = 0
        tree_lstm_memory = {}
        self.sqlt = sqlt
        self.sql = self.sqlt.sql

        self.aliasname2fullname = {}
        self.runner = runner
        self.device = device
        self.aliasname2fromtable={}

        format = os.environ['RTOS_GV_FORMAT'] if os.environ.get('RTOS_GV_FORMAT') is not None else 'svg'
        self.join_tree_repr = gv.Digraph(format=format, graph_attr={"rankdir": "TB"})

        logging.debug(f"\nFile name: {self.sqlt.filename}\nSQL: {self.sql}")

        if isinstance(runner, PGRunner):
            parse_result = parse_dict(self.sql)[0]["SelectStmt"]
            self.target_table_list = [TargetTableSQL(x["ResTarget"]) for x in parse_result["targetList"]]
            self.from_table_list = [FromTableSQL(x["RangeVar"]) for x in parse_result["fromClause"]]
       
            for table in self.from_table_list:
                self.aliasname2fromtable[table.getAliasName()] = table
                self.aliasname2fullname[table.getAliasName()] = table.getFullName()

            logging.debug(f"There are {len(parse_result)} items, {len(self.from_table_list)} from, {len(self.target_table_list)} target")

            self.aliasnames = setlist(self.aliasname2fromtable.keys())
            self.comparison_list =[ComparisonSQL(x) for x in parse_result["whereClause"]["BoolExpr"]["args"]]
            self.aliasnames_root_set = setlist([x.getAliasName() for x in self.from_table_list])

            logging.debug(self.comparison_list)

        elif isinstance(runner, ISQLRunner):
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

            def get_joins(tps: List[Dict[str, str]]) -> Dict[str, JoinISQL]:
                equal_comparisons = {}

                subjects: List[str] = []
                predicates: List[str] = []
                objects: List[str] = []
                for tp in tps:
                    s, p, o = tp['subject'], tp['predicate'], tp['object']
                    subjects.append(s)
                    predicates.append(p)
                    objects.append(o)
                
                for i, (s_i, p_i, o_i) in enumerate(zip(subjects, predicates, objects)):
                    for j, (s_j, p_j, o_j) in enumerate(zip(subjects, predicates, objects)):
                        if i == j: continue

                        """There are 4 type of joins in sparql
                        subject-object (path join): 
                            - form: "(?s1 ?p1 ?o1). (?s2 ?p2 ?s1)"
                        subject-subject (star join): 
                            - form: "(?s1 ?p1 ?o1). (?s1 ?p2 ?o2)"
                        object-object (star join): 
                            - form: "(?s1 ?p1 ?o1). (?s2 ?p2 ?o1)"
                        object-subject (path join): 
                            - form: "(?s1 ?p1 ?o1). (?o1 ?p2 ?s2)"
                        """

                        if s_i == o_j:
                            # from_table: fromOther fromTable joinCol
                            # target_table: joinCol targetTable targetOther
                            equal_comparisons[f"{p_j} |><| {p_i} ({ o_j })"] = JoinISQL({
                                "fromTable": p_j,
                                "targetTable": p_i,
                                "joinCol": s_i,
                                "fromOtherCol": s_j,
                                "targetOtherCol": o_i,
                                "type": "so"
                            })

                        if s_i == s_j:
                            # from_table: joinCol fromTable fromOther
                            # target_table: joinCol targetTable targetOther
                            equal_comparisons[f"{p_i} |><| {p_j} ({ s_i })"] = JoinISQL({
                                "fromTable": p_i,
                                "targetTable": p_j,
                                "joinCol": s_j,
                                "fromOtherCol": o_i,
                                "targetOtherCol": o_j,
                                "type": "ss"
                            })

                            equal_comparisons[f"{p_j} |><| {p_i} ({ s_j })"] = JoinISQL({
                                "fromTable": p_j,
                                "targetTable": p_i,
                                "joinCol": s_i,
                                "fromOtherCol": o_j,
                                "targetOtherCol": o_i,
                                "type": "ss"
                            })

                        if o_i == o_j and o_i.startswith("?"):
                            # from_table: fromOther fromTable joinCol
                            # target_table: targetOther targetTable joinCol
                            equal_comparisons[f"{p_i} |><| {p_j} ({ o_i })"] = JoinISQL({
                                "fromTable": p_i,
                                "targetTable": p_j,
                                "joinCol": o_i,
                                "fromOtherCol": s_i,
                                "targetOtherCol": s_j,
                                "type": "oo"
                            })

                            equal_comparisons[f"{p_j} |><| {p_i} ({ o_j })"] = JoinISQL({
                                "fromTable": p_j,
                                "targetTable": p_i,
                                "joinCol": o_j,
                                "fromOtherCol": s_j,
                                "targetOtherCol": s_i,
                                "type": "oo"
                            })
                        
                        if o_i == s_j:
                            # from_table: fromOther fromTable joinCol
                            # target_table: joinCol targetTable targetOther
                            equal_comparisons[f"{p_i} |><| {p_j} ({ o_i })"] = JoinISQL({
                                "fromTable": p_i,
                                "targetTable": p_j,
                                "joinCol": o_i,
                                "fromOtherCol": s_i,
                                "targetOtherCol": o_j,
                                "type": "os"
                            })
                    
                return equal_comparisons

            parse_result: ParsedQuery = QueryParser.parse(self.sql)
            #logging.debug(parse_result.triple_patterns)
            #logging.debug(parse_result.filters)
            self.all_joins_list = get_joins(parse_result.triple_patterns)
            
            self.from_table_list: SetList[FromTableISQL] = setlist(map(lambda join: join.getFromTable(), self.all_joins_list.values()))
            self.target_table_list: SetList[TargetTableISQL] = setlist(map(lambda join: join.getTargetTable(), self.all_joins_list.values()))
            
            self.all_table_list: SetList[DummyTableISQL] = self.from_table_list | self.target_table_list
            logging.debug(f"There are {len(parse_result.triple_patterns)} triple patterns, {len(self.from_table_list)} from, {len(self.target_table_list)} target")
            
            tp_set = setlist(map(lambda tp: f'{tp["subject"]} <{tp["predicate"]}> {tp["object"]}', parse_result.triple_patterns))
            all_table_set = setlist(map(lambda table: str(table), self.all_table_list))
            if len(all_table_set - tp_set) != 0:
                raise ValueError(f"The following triple pattern does not exist: {all_table_set - tp_set}")

            for table in self.all_table_list :
                self.aliasname2fromtable[table.getAliasName()] = table
                self.aliasname2fullname[table.getAliasName()] = table.getFullName()
                db_info.name2table[table.getFullName()].updateTable(table)

            self.aliasnames = setlist(self.aliasname2fromtable.keys())
            self.comparison_list = list()
            self.comparison_list.extend([ComparisonISQL(x) for x in parse_result.filters])
            logging.debug(f"There are {len(self.comparison_list)}/{len(parse_result.filters)} filter items: {self.comparison_list}")

            # Add equal comparison, albeit to a literal or a variable (join)
            for join_name, join in self.all_joins_list.items():
                eq_comp = ComparisonISQLEqual(join_name, join).breakdown()
                self.comparison_list.extend(eq_comp)

            logging.debug(f"There are {len(self.comparison_list)} comparisons")
            logging.debug('\n'.join(map(str, self.comparison_list)))

            self.aliasnames_root_set = setlist([x.getAliasName() for x in self.all_table_list])
        else:
            raise NotImplementedError(f"runner must be instance of {DBRunner}")

        self.db_info = db_info
        self.join_list: Dict[ str, List[Tuple[str, str]] ] = {}
        self.filter_list: Dict[str, List[ComparisonISQL]] = {}

        """ The join tree has dual presentation, one for the name and one for alias. Both tree are updated at the same time in the same way.
        """

        self.aliasnames_fa = {}
        self.aliasnames_set: Dict[str, setlist] = {}
        self.aliasnames_join_set: Dict[str, setlist] = {}
        self.left_son: Dict[int, str] = {}
        self.right_son: Dict[int, str] = {}
        self.total = 0
        self.left_aliasname: Dict[int, str] = {}
        self.right_aliasname: Dict[int, str] = {}

        """
        Initialize feature set F
        size = max_column_in_table * 2 columns + 1 subject feature + 1 object feature
        """
        self.table_fea_set = {}
        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = [0.0]*(config.max_column_in_table * NB_FEATURE_SLOTS)
            self.join_list[aliasname] = []

        self.join_candidate: Set[Tuple[str, str]] = setlist()
        
        # Initialize M matrix

        self.join_matrix = []
        for idx in range(len(self.db_info)):
            self.join_matrix.append([0]*len(self.db_info))
            
        for comparison in self.comparison_list:
            """In SQL:
            - For each comparison t1.c1 = t2.c2 do:

            """

            # When the aliases are present on 2 operands, 
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

                self.table_fea_set[left_aliasname][table_idx * NB_FEATURE_SLOTS] = 1

                if os.environ['RTOS_ENGINE'] == "sparql":
                    self.table_fea_set[left_aliasname][table_idx * NB_FEATURE_SLOTS + 2] = 1

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
                self.table_fea_set[right_aliasname][table_idx * NB_FEATURE_SLOTS] = 1
                
                if os.environ['RTOS_ENGINE'] == "sparql":
                    self.table_fea_set[right_aliasname][table_idx * NB_FEATURE_SLOTS + 3] = 1


                # Add two join candidate left-right then right-left. While they yield the same result but one is faster than the another.
                self.join_candidate.add((left_aliasname,right_aliasname))
                self.join_candidate.add((right_aliasname,left_aliasname))
                
                # Update M matrix
                idx0 = self.db_info.name2idx[left_fullname]
                idx1 = self.db_info.name2idx[right_fullname]
                self.join_matrix[idx0][idx1] = 1
                self.join_matrix[idx1][idx0] = 1
            else: 
                left_aliasname = comparison.aliasname_list[0]
                    
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
                selectivity = self.runner.getSelectivity(
                    str(self.aliasname2fromtable[left_aliasname]), 
                    comparison.toString()
                )
                self.table_fea_set[left_aliasname][table_idx * NB_FEATURE_SLOTS + 1] += selectivity

                if os.environ['RTOS_ENGINE'] == "sparql":
                    self.table_fea_set[left_aliasname][table_idx * NB_FEATURE_SLOTS + 2] += selectivity

        for aliasname in self.aliasnames_root_set:
            self.table_fea_set[aliasname] = torch.tensor(self.table_fea_set[aliasname],device = self.device).reshape(1,-1).detach()
            self.aliasnames_set[aliasname] = setlist([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = setlist()
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

        if os.environ['RTOS_ENGINE'] == "sql":
            self.aliasnames_root_set = setlist([x.getAliasName() for x in self.from_table_list])
        else:
            self.aliasnames_root_set = setlist([x.getAliasName() for x in self.all_table_list])

        self.left_aliasname  = {}
        self.right_aliasname =  {}
        self.aliasnames_join_set = {}
        for aliasname in self.aliasnames_root_set:
            self.aliasnames_set[aliasname] = setlist([aliasname])
            for y in self.join_list[aliasname]:
                if aliasname not in self.aliasnames_join_set:
                    self.aliasnames_join_set[aliasname] = setlist()
                self.aliasnames_join_set[aliasname].add(y[0])

        self.total = 0

    def findFather(self,node_name: str) -> str:
        """Retrieve the farthest ascendant of the given node

        Args:
            node_name (str): [description]

        Returns:
            str: [description]
        """
        fa_name = node_name
        while fa_name in self.aliasnames_fa.keys():
            fa_name = self.aliasnames_fa[fa_name]
        while node_name in self.aliasnames_fa.keys():
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

    def recTableISQL(self,node: Union[int, str]) -> Union[int, List[str]]:
        """Recursively construct sparql query from a given node.
        Args:
            node (Union[int, str]): a position in the tree or the join candidate

        Returns:
            Union[int, List[str]]: list of elected candidates or a pointer to the next batch of candidates
        """

        # Node is int means that it's a root node
        if isinstance(node,int):

            res = setlist()

            left_son = self.left_son[node]
            right_son = self.right_son[node]

            self.join_tree_repr.node(str(hash(node)), str(node))
            self.join_tree_repr.node(str(hash(left_son)), str(left_son))
            self.join_tree_repr.node(str(hash(right_son)), str(right_son))

            leftRes = self.recTableISQL(left_son)
            self.join_tree_repr.edge(str(hash(node)), str(hash(left_son)))
            res.update(leftRes)
            
            filter_list = []
            on_list = []
            if left_son in self.filter_list.keys():
                for condition in self.filter_list[left_son]:
                    filter_list.append(condition.toString())
        
            if right_son in self.filter_list.keys() :
                for condition in self.filter_list[right_son]:
                    filter_list.append(condition.toString())

            cpList = []
            joined_aliasname = setlist([self.left_aliasname[node],self.right_aliasname[node]])
            for left_table in self.aliasnames_set[left_son]:
                for right_table, comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[right_son]:
                        left_aliasname, right_aliasname = comparison.aliasname_list[0], comparison.aliasname_list[1]
                        if ( left_aliasname in joined_aliasname and right_aliasname in joined_aliasname):
                            cpList.append(comparison.toString())
                        else:
                            on_list.append(comparison.toString())
            
            rightRes = self.recTableISQL(right_son)
            self.join_tree_repr.edge(str(hash(node)), str(hash(right_son)))
            res.update(rightRes)

            # inner join
            if len(filter_list + on_list + cpList) > 0:
                res.update(cpList + on_list + filter_list)

            return res
        else:
            return [str(self.aliasname2fromtable[node])]

    def recTableSQL(self,node):
        if isinstance(node,int):

            left_son = self.left_son[node]
            right_son = self.right_son[node]

            res =  "("
            leftRes = self.recTableSQL(left_son)
            if not left_son in self.aliasnames:
                leftRes = leftRes[1:-1]

            self.join_tree_repr.node(str(hash(node)), str(node))
            self.join_tree_repr.node(str(hash(left_son)), str(left_son))
            self.join_tree_repr.node(str(hash(right_son)), str(right_son))

            self.join_tree_repr.edge(str(hash(node)), str(hash(left_son)))

            res += leftRes + "\n"
            filter_list = []
            on_list = []
            if left_son in self.filter_list:
                for condition in self.filter_list[left_son]:
                    filter_list.append(str(condition))

            if right_son in self.filter_list :
                for condition in self.filter_list[right_son]:
                    filter_list.append(str(condition))

            cpList = []
            joined_aliasname = setlist([self.left_aliasname[node],self.right_aliasname[node]])
            for left_table in self.aliasnames_set[left_son]:
                for right_table,comparison in self.join_list[left_table]:
                    if right_table in self.aliasnames_set[right_son]:
                        left_aliasname, right_aliasname = comparison.aliasname_list[0], comparison.aliasname_list[1]
                        if ( left_aliasname in joined_aliasname and right_aliasname in joined_aliasname):
                            cpList.append(str(comparison))
                        else:
                            on_list.append(str(comparison))

            rightRes = self.recTableSQL(right_son)
            
            if len(filter_list+on_list+cpList)>0:
                res += "inner join "
                res += rightRes
                res += "\non "
                res += " AND ".join(cpList + on_list + filter_list)
            else:
                res += "cross join "
                res += rightRes

            self.join_tree_repr.edge(str(hash(node)), str(hash(right_son)))

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

    def encode_tree_fold(self,fold: Fold, node_idx: int):
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

    def toSql(self) -> str:
        """Convert this join tree to executable query

        Returns:
            str: executable query string for in SQL or in SPARQL
        """
        root = self.total - 1
        logging.debug(f"Root: {root}")

        if os.environ["RTOS_ENGINE"] == "sql":
            res = "select "+",\n".join([str(x) for x in self.target_table_list])+"\n"
            res += "from " + self.recTableSQL(root)[1:-1]
            res += ";"
        else:
            res = "SELECT * WHERE { \n\t" 
            res += ' .\n\t'.join(self.recTableISQL(root))
            res += "\n};"

        # Graphviz
        # fn = os.path.basename(self.sqlt.filename).split('.')[0]
        # self.join_tree_repr.render(os.path.join(config.JOBDir, fn, f"{fn}_{hash(self)}.gv"))

        logging.debug("Proposed plan:")
        logging.debug(res)

        return res

    def plan2Cost(self):
        self.proposedPlan = self.toSql()
        return self.runner.getLatency(self.sqlt, self.proposedPlan, force_order=True)

    @property
    def plan(self):
        return self.proposedPlan





