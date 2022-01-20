import logging
import os
import re
from typing import Dict, List, OrderedDict, Tuple, Union
from collections_extended.setlists import SetList
import numpy as np
from rdflib.plugins.sparql.operators import Builtin_REGEX, Builtin_STR, ConditionalAndExpression, ConditionalOrExpression, RelationalExpression, UnaryMinus, UnaryNot, UnaryPlus
from torch._C import BoolType, Value
from rdflib.term import Literal, Variable, URIRef
from rdflib.plugins.sparql import parserutils
from torch.functional import einsum
import yaml
from collections_extended import setlist

class TargetTableSQL:
    def __init__(self, target):
        """
        {'location': 7, 'name': 'alternative_name', 'val': {'FuncCall': {'funcname': [{'String': {'str': 'min'}}], 'args': [{'ColumnRef': {'fields': [{'String': {'str': 'an'}}, {'String': {'str': 'name'}}], 'location': 11}}], 'location': 7}}}
        """
        self.target = target
    #         logging.debug(self.target)

    def getValue(self,):
        columnRef = self.target["val"]["FuncCall"]["args"][0]["ColumnRef"]["fields"]
        return columnRef[0]["String"]["str"]+"."+columnRef[1]["String"]["str"]

    def __str__(self,):
        try:
            return self.target["val"]["FuncCall"]["funcname"][0]["String"]["str"]+"(" + self.getValue() + ")" + " AS " + self.target['name']
        except:
            if "FuncCall" in self.target["val"]:
                return "count(*)"
            else:
                return "*"

class DummyTableISQL:
    def __init__(self, s, p, o) -> None:
        self._s = s
        self._p = p
        self._o = o
        
        fn_s = re.sub(r'\d', '', s) if s.startswith("?") else s
        fn_o = re.sub(r'\d', '', o) if o.startswith("?") else o

        self.alias = f"{s} <{p}> {o}"
        self.fullname = f"{fn_s} <{p}> {fn_o}"
    
    @property
    def s(self) -> str:
        return self._s

    @property
    def p(self) -> str:
        return self._p
    
    @property
    def o(self) -> str:
        return self._o

    def spo(self):
        return self._s, self._p, self._o

    def getAliasName(self):
        return self.alias
    
    def getFullName(self):
        return self._p

    def __str__(self):
        return self.getAliasName()
    
    def __eq__(self, __o: object) -> bool:
        return self.alias == __o.alias
    
    def __hash__(self) -> int:
        return hash(self.alias)

class TargetTableISQL(DummyTableISQL):

    def __init__(self, s, p, o) -> None:
        super().__init__(s, p, o)

    def getValue(self):
        return self.getFullName()
    

class FromTableSQL:
    def __init__(self, from_table):
        """
        {'alias': {'Alias': {'aliasname': 'an'}}, 'location': 168, 'inhOpt': 2, 'relpersistence': 'p', 'relname': 'aka_name'}
        """
        self.from_table = from_table

    def getFullName(self,):
        return self.from_table["relname"]

    def getAliasName(self,):
        return self.from_table["alias"]["Alias"]["aliasname"]

    def __str__(self,):
        return self.getFullName()+" AS "+self.getAliasName()

class FromTableISQL(DummyTableISQL):
    def __init__(self, s, p, o) -> None:
        super().__init__(s, p, o)

class ExprISQL:
    def __init__(self, expr: parserutils.Expr, list_kind = 0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0

    def isCol(self):
        return not isinstance(self.expr, Literal)
        #return True

    def getAliasName(self):
        if isinstance(self.expr, Literal) or isinstance(self.expr, Variable):
            return self.getValue(self.expr)
        elif self.expr._evalfn.__name__ == Builtin_REGEX.__name__ :
            var, _ = self.getValue(self.expr)
            return var
        elif self.expr._evalfn.__name__ == Builtin_STR.__name__ :
            return self.getValue(self.expr)
        else:
            raise NotImplementedError(f"No Known type of {self.expr}")
        
    def getColumnName(self):
        # if isinstance(self.expr, Literal):
        #     return self.getAliasName()
        # else:
        #     return re.sub(r'\d', '', self.getAliasName())
        return self.getAliasName()
        

    def getValue(self, value_expr: parserutils.Expr):
        if isinstance(value_expr, Literal):
            if value_expr.datatype == URIRef('http://www.w3.org/2001/XMLSchema#string'):
                return value_expr.value
            elif value_expr.datatype == URIRef('http://www.w3.org/2001/XMLSchema#integer'):
                self.isInt = True
                self.val = value_expr.value
                return str(self.val)
            else:
                return repr(value_expr.value)
        elif isinstance(value_expr, (Variable, str)):
            return f'?{str(value_expr)}'
        elif value_expr._evalfn.__name__ == Builtin_REGEX.__name__:
            return self.getValue(value_expr.get('text')), self.getValue(value_expr.get('pattern'))
        elif value_expr._evalfn.__name__ == Builtin_STR.__name__:
            return self.getValue(value_expr.get('arg'))
        else:
            raise NotImplementedError(f"Unknown expression of type {value_expr}")

    def __str__(self,):
        if isinstance(self.expr, (Variable, Literal)):
            return self.getValue(self.expr)
        elif self.expr._evalfn.__name__ == Builtin_REGEX.__name__ :
            var, pattern = self.getValue(self.expr)
            return f'regex({var}, {pattern})'
        elif self.expr._evalfn.__name__ == Builtin_STR.__name__ :
            return f'str({self.getValue(self.expr)})'
        else:
            raise NotImplementedError(f"Unknown expression of type {self.expr}")

    def get_variables(self) -> SetList:
        res = setlist()
        if isinstance(self.expr, Variable):
            res.append(f'?{str(self.expr)}')
        elif isinstance(self.expr, Literal):
            pass
        elif self.expr._evalfn.__name__ == Builtin_REGEX.__name__:
            res.append(self.getValue(self.expr.get('text')))
        elif self.expr._evalfn.__name__ == Builtin_STR.__name__:
            res.append(self.getValue(self.expr.get('arg')))
        else:
            raise NotImplementedError(f"Unknown expression of type {self.expr}")

        return res

class JoinISQL:
    def __init__(self, join) -> None:
        self.fromTable = join['fromTable']
        self.targetTable = join['targetTable']
        self.joinCol = join['joinCol']
        self.fromOtherCol = join['fromOtherCol']
        self.targetOtherCol = join['targetOtherCol']
        self.type = join['type']

        self.fromJoin, self.targetJoin = None, None

        # Star joins
        if self.type == "so":
            self.fromJoin = FromTableISQL(self.fromOtherCol, self.fromTable, self.joinCol)
            self.targetJoin = TargetTableISQL(self.joinCol, self.targetTable, self.targetOtherCol)

        elif self.type == "os":
            self.fromJoin = FromTableISQL(self.fromOtherCol, self.fromTable, self.joinCol)
            self.targetJoin = TargetTableISQL(self.joinCol, self.targetTable, self.targetOtherCol)

        # Path join
        elif self.type == "ss":
            self.fromJoin = FromTableISQL(self.joinCol, self.fromTable, self.fromOtherCol)
            self.targetJoin = TargetTableISQL(self.joinCol, self.targetTable, self.targetOtherCol)

        elif self.type == "oo":
            self.fromJoin = FromTableISQL(self.fromOtherCol, self.fromTable, self.joinCol)
            self.targetJoin = TargetTableISQL(self.targetOtherCol, self.targetTable, self.joinCol)
                
    def getFromTable(self) -> FromTableISQL:
        return self.fromJoin
    
    def getTargetTable(self) -> TargetTableISQL:
        return self.targetJoin

class ExprSQL:
    def __init__(self, expr,list_kind = 0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0
    def isCol(self,):
        return isinstance(self.expr, dict) and "ColumnRef" in self.expr

    def getValue(self, value_expr):
        if "A_Const" in value_expr:
            value = value_expr["A_Const"]["val"]
            if "String" in value:
                return "'" + value["String"]["str"]+"\'"
            elif "Integer" in value:
                self.isInt = True
                self.val = value["Integer"]["ival"]
                return str(value["Integer"]["ival"])
            else:
                raise "unknown Value in Expr"
        elif "TypeCast" in value_expr:
            if len(value_expr["TypeCast"]['typeName']['TypeName']['names'])==1:
                return value_expr["TypeCast"]['typeName']['TypeName']['names'][0]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+"'"
            else:
                if value_expr["TypeCast"]['typeName']['TypeName']['typmods'][0]['A_Const']['val']['Integer']['ival']==2:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' month"
                else:
                    return value_expr["TypeCast"]['typeName']['TypeName']['names'][1]['String']['str']+" '"+value_expr["TypeCast"]['arg']['A_Const']['val']['String']['str']+ "' year"
        else:
            raise "unknown Value in Expr"

    def getAliasName(self,):
        return self.expr["ColumnRef"]["fields"][0]["String"]["str"]

    def getColumnName(self,):
        return self.expr["ColumnRef"]["fields"][1]["String"]["str"]

    def __str__(self,):
        if self.isCol():
            return self.getAliasName()+"."+self.getColumnName()
        elif isinstance(self.expr, dict) and "A_Const" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, dict) and "TypeCast" in self.expr:
            return self.getValue(self.expr)
        elif isinstance(self.expr, list):
            if self.list_kind == 6:
                return "("+",\n".join([self.getValue(x) for x in self.expr])+")"
            elif self.list_kind == 10:
                return " AND ".join([self.getValue(x) for x in self.expr])
            else:
                raise "list kind error"

        else:
            raise "No Known type of Expr"

class ValuesISQL:
    def __init__(self, values: List[Dict[str, str]]) -> None:
        self._mappings = values
        self._variables = setlist()
        for mapping in values:
            self._variables.update(mapping.keys())
    
    def __str__(self) -> str:
        res = f'VALUES ({" ".join(self._variables)})' + ' {'
        res += " ".join([ f'({" ".join(v.values())})' for v in self._mappings ])
        res += ' }'
        return res
    
    @property
    def variables(self):
        return self._variables

    @property
    def mappings(self):
        return self._mappings

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return hash(self.__str__())
    
    def __eq__(self, __o: object) -> bool:
        return self.mappings == __o.mappings

class ComparisonDummy():
    def __init__(self, s, p, o, cl, al) -> None:
        self.s = s
        self.p = p
        self.o = o

        self.column_list = cl
        self.aliasname_list = al

    def toString(self):
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, __o: object) -> bool:
        return self.__str__() == __o.__str__()

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __str__(self) -> str:
        return f"{self.s} <{self.p}> {self.o}"

class ComparisonISQLEqual:
    """Build equivalent of t1.c1 = t2.c2 for sparql

    Case 1: Join column
    (?s1 ?p1 ?o). (?o ?p2 ?o2) is same as p1.o = p2.o
    (?s ?p1 ?o1). (?s ?p2 ?o2) is same as p1.s = p2.s
    (?s1 ?p1 ?o). (?s2 ?p2 ?o) is same as p1.o = p2.o
    (?o ?p1 ?o1). (?s2 ?p2 ?o2) is same as p1.o = p2.o

    Case 2: equal comparison
    (?o1 ?p1 "constant") is same as p1.o1 = "constant"
    """        
    def __init__(self, join_name: str, join: JoinISQL) -> None:
        self.name = join_name
        self.from_table = join.getFromTable()
        self.target_table = join.getTargetTable()

        self.column_list = [
            join.fromOtherCol,
            join.targetOtherCol
        ]

        self.aliasname_list = [
            self.from_table.getAliasName(),
            self.target_table.getAliasName()
        ]

    def breakdown(self):
        return (
            ComparisonDummy(*self.from_table.spo(), self.column_list, self.aliasname_list),
            ComparisonDummy(*self.target_table.spo(), self.column_list, self.aliasname_list)
        )

    def __eq__(self, __o: object) -> bool:
        return str(self) == str(__o)

    def __hash__(self) -> int:
        return sum(map(hash, self.breakdown()))

    def __str__(self) -> str:
        return str(tuple(map(str, self.breakdown())))

    def toString(self) -> str:
        res = " . ".join(map(lambda x: x.toString(), self.breakdown()))
        return res
        
class ComparisonISQL:
    def __init__(self, comparison) -> None:

        def get_op(comp: parserutils.Expr):
            if comp._evalfn.__name__ == UnaryNot.__name__:
                return "!"
            elif comp._evalfn.__name__ == UnaryMinus.__name__:
                return "-"
            elif comp._evalfn.__name__ == UnaryPlus.__name__:
                return "+"
            elif comp._evalfn.__name__ == ConditionalAndExpression.__name__:
                return "&&"
            elif comp._evalfn.__name__ == ConditionalOrExpression.__name__:
                return "||"
            else:
                raise ValueError(f"Did not recognize operator {comp._evalfn.__name__}")

        """
        A filter looks like any of these:
            [
                RelationalExpression_{
                    'expr': rdflib.term.Variable('t_production_year'), 
                    'op': '<=', 
                    'other': rdflib.term.Literal('2010', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')), 
                    '_vars': set()
                }, 
                UnaryNot_{
                    'expr': Builtin_REGEX_{
                        'text': rdflib.term.Variable('mc_note'), 
                        'pattern': rdflib.term.Literal('\\(as Metro-Goldwyn-Mayer Pictures\\)'), 
                        '_vars': {rdflib.term.Variable('mc_note')}
                    }, 
                    '_vars': {rdflib.term.Variable('mc_note')}
                }, 
                RelationalExpression_{
                    'expr': rdflib.term.Variable('t_production_year'), 
                    'op': '>=', 
                    'other': rdflib.term.Literal('2005', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')), 
                    '_vars': set()
                }
            ]
        """
        self.comparison: parserutils.Expr = comparison
        self.column_list = []
        self.comp_list: List[Union[ComparisonISQL, ComparisonISQLEqual, ComparisonDummy]] = []
        self.aliasname_list = []
        self.kind = None
        self.op = None
        self.lexpr = None
        self.rexpr = None

        #logging.debug(self.comparison, type(self.comparison))

        def update_alias_and_column(alias, column):
            if isinstance(alias, list):
                self.aliasname_list.extend(alias)
            else: 
                self.aliasname_list.append(alias)

            if isinstance(column, list):
                self.column_list.extend(column)
            else: 
                self.column_list.append(column)

        if isinstance(self.comparison, (Variable, Literal)):
            self.lexpr = ExprISQL(self.comparison)
            self.aliasname_list.append(self.lexpr.getAliasName())
            self.column_list.append(self.lexpr.getColumnName())
            
        elif self.comparison._evalfn.__name__ == RelationalExpression.__name__:
            self.kind = 0

            self.lexpr = ComparisonISQL(self.comparison.get("expr"))
            self.op = self.comparison.get("op")
            self.rexpr = ComparisonISQL(self.comparison.get("other"))

            if self.lexpr.isCol():
                update_alias_and_column(self.lexpr.getAliasName(), self.lexpr.getColumnName())

            if self.rexpr.isCol():
                update_alias_and_column(self.rexpr.getAliasName(), self.rexpr.getColumnName())

            self.comp_kind = 0

        elif "Unary" in self.comparison._evalfn.__name__:
            self.lexpr = ComparisonISQL(self.comparison.get('expr'))
            self.kind = 1
            self.op = get_op(self.comparison)

            if self.lexpr.isCol():
                update_alias_and_column(self.lexpr.getAliasName(), self.lexpr.getColumnName())

            self.comp_kind = 2

        elif "Conditional" in self.comparison._evalfn.__name__:
            """
            ConditionalOrExpression_{
                'expr': RelationalExpression_{
                    'expr': rdflib.term.Variable('t_production_year'), 
                    'op': '>=', 
                    'other': rdflib.term.Literal('2005', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')), 
                    '_vars': set()
                }, 
                'other': [
                    RelationalExpression_{
                        'expr': rdflib.term.Variable('t_production_year'), 
                        'op': '<=', 
                        'other': rdflib.term.Literal('2010', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')), 
                        '_vars': set()
                    }
                ], 
                '_vars': set()
            }
            """

            # "boolop"
            self.kind = 1
            self.op = get_op(self.comparison)
            self.lexpr = ComparisonISQL(self.comparison.get('expr'))
            update_alias_and_column(self.lexpr.getAliasName(), self.lexpr.getColumnName())
            self.comp_list.append(self.lexpr)
            self.comp_list.extend([ComparisonISQL(x) for x in self.comparison.get('other')])

            self.comp_kind = 2
        
        elif self.comparison._evalfn.__name__ == Builtin_REGEX.__name__:
            self.kind = 7

            self.lexpr = ExprISQL(self.comparison)
            update_alias_and_column(self.lexpr.getAliasName(), self.lexpr.getColumnName())


        elif self.comparison._evalfn.__name__ == Builtin_STR.__name__:
            self.lexpr = ComparisonISQL(self.comparison.get('arg'))
            update_alias_and_column(self.lexpr.getAliasName(), self.lexpr.getColumnName())

        else:
            raise NotImplementedError(f"No handler for {self.comparison.name}, value: \n {self.comparison}")

    def isCol(self):
        return not isinstance(self.comparison, Literal)
        #return True

    def getAliasName(self) -> List[str]:
        res = []
        
        if len(self.comp_list) == 0:
            if self.lexpr is not None:
                alias = self.lexpr.getAliasName()
                if isinstance(alias, list): res.extend(alias) 
                else: res.append(alias)
            if self.rexpr is not None:
                alias = self.rexpr.getAliasName()
                if isinstance(alias, list): res.extend(alias) 
                else: res.append(alias)
        else:
            for comp in self.comp_list:
                res.extend(comp.getAliasName())

        #return res
        return [ r for r in res if r.startswith('?') ]

    def getColumnName(self) -> List[str]:
        res = []
        
        if len(self.comp_list) == 0:
            if self.lexpr is not None:
                alias = self.lexpr.getColumnName()
                if isinstance(alias, list): res.extend(alias) 
                else: res.append(alias)
            if self.rexpr is not None:
                alias = self.rexpr.getColumnName()
                if isinstance(alias, list): res.extend(alias) 
                else: res.append(alias)
        else:
            for comp in self.comp_list:
                res.extend(comp.getColumnName())
        
        return [ r for r in res if r.startswith('?') ]

    def toString(self) -> str:
        return f'FILTER({ str(self) })'

    def __eq__(self, __o: object) -> bool:
        return self.__str__() == __o.__str__()

    def __hash__(self) -> int:
        return hash(self.__str__())

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self):
        res = ""
        if len(self.comp_list) == 0:
            if isinstance(self.comparison, (Variable, Literal)):
                res += str(self.lexpr)
            elif "Relational" in self.comparison.name or "Conditional" in self.comparison.name:
                res += f'{ str(self.lexpr) } { self.op } { str(self.rexpr) }'
            elif "Unary" in self.comparison.name:
                res += f'{ self.op }{ str(self.lexpr) }'
            elif self.comparison.name == Builtin_REGEX.__name__:
                res += str(self.lexpr)
            elif self.comparison.name == Builtin_STR.__name__:
                res += f"str({str(self.lexpr)})"
            else:
                raise NotImplementedError(f"Unknown handler for type {self.comparison.name}")
        else:
            res += f' {self.op} '.join([ str(comp) for comp in self.comp_list ])
        
        return res

    def get_variables(self) -> SetList:
        res = []
        if len(self.comp_list) == 0:
            if isinstance(self.comparison, Variable):
                res.append(str(self.lexpr))
            else:
                if self.lexpr is not None:
                    res.extend(self.lexpr.get_variables())
                if self.rexpr is not None:
                    res.extend(self.rexpr.get_variables())
        else:
            for comp in self.comp_list:
                res.extend(comp.get_variables())
        return setlist(res)

class ComparisonSQL:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = ExprSQL(comparison["A_Expr"]["lexpr"])
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = ExprSQL(comparison["A_Expr"]["rexpr"],self.kind)
            else:
                self.rexpr = ComparisonSQL(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = ExprSQL(comparison["NullTest"]["arg"])
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            # "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [ComparisonSQL(x) for x in comparison["BoolExpr"]["args"]]
            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.aliasname_list.append(comp.lexpr.getAliasName())
                    self.lexpr = comp.lexpr
                    self.column_list.append(comp.lexpr.getColumnName())
                    break
            self.comp_kind = 2
    def isCol(self,):
        return False

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return self.__str__()
    
    def __str__(self,):

        if self.comp_kind == 0:
            Op = ""
            if self.kind == 0:
                Op = self.comparison["A_Expr"]["name"][0]["String"]["str"]
            elif self.kind == 7:
                if self.comparison["A_Expr"]["name"][0]["String"]["str"]=="!~~":
                    Op = "not like"
                else:
                    Op = "like"
            elif self.kind == 6:
                Op = "IN"
            elif self.kind == 10:
                Op = "BETWEEN"
            else:
                import json
                logging.debug(str(json.dumps(self.comparison, sort_keys=True, indent=4)))
                raise "Operation ERROR"
            return str(self.lexpr)+" "+Op+" "+str(self.rexpr)
        elif self.comp_kind == 1:
            if self.kind == 1:
                return str(self.lexpr)+" IS NOT NULL"
            else:
                return str(self.lexpr)+" IS NULL"
        else:
            res = ""
            for comp in self.comp_list:
                if res == "":
                    res += "( "+str(comp)
                else:
                    if self.kind == 1:
                        res += " OR "
                    else:
                        res += " AND "
                    res += str(comp)
            res += ")"
            return res

class TableSQL:
    def __init__(self, table_tree):
        self.name = table_tree["relation"]["RangeVar"]["relname"]
        self.column2idx = {}
        self.idx2column = {}
        for idx, columndef in enumerate(table_tree["tableElts"]):
            self.column2idx[columndef["ColumnDef"]["colname"]] = idx
            self.idx2column[idx] = columndef["ColumnDef"]["colname"]

    def __repr__(self) -> str:
        return f"Name: {self.name}, col2idx: {self.column2idx}, idx2col: {self.idx2column}"

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))

class TableISQL:
    def __init__(self, table_name) -> None:
        self.name = table_name
        self.column2idx = {}
        self.idx2column = {}

    def updateTable(self, table: Union[TargetTableISQL, FromTableISQL]):
        cols = [ table.s, table.o ]
        for col in cols:
            if col not in self.column2idx.keys():
                idx = len(self.column2idx)
                self.column2idx[col] = idx
                self.idx2column[idx] = col
    
    def __repr__(self) -> str:
        return f"Name: {self.name}, col2idx: {self.column2idx}, idx2col: {self.idx2column}"

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))
        
class DB:
    def __init__(self, schema: str, config: dict):

        self.config = config
        
        from psqlparse import parse_dict
        parse_tree = parse_dict(schema)

        self.tables: List[Union[TableISQL, TableSQL]] = []
        self.name2idx: Tuple[str, int] = {}
        self.table_names: List[str] = []
        self.name2table: Tuple[ str, Union[TableISQL, TableSQL] ] = {}
        self.size = 0
        self.TREE_NUM_IN_NET = config["database"]["tree_num_in_net"]

        def add_table(table: Union[TableSQL, TableISQL], idx):
            self.tables.append(table)
            self.table_names.append(table.name)
            self.name2idx[table.name] = idx
            self.name2table[table.name] = table

        if config["database"]["engine_class"] == "sql":
            for idx, table_tree in enumerate(parse_tree):
                table_name = table_tree["CreateStmt"]["relation"]["RangeVar"]["relname"]
                add_table(TableSQL(table_tree["CreateStmt"]), idx)

        elif config["database"]["engine_class"] == "sparql":
            relations = open(os.path.join(config["database"]["JOBDir"], 'relations.txt'), 'r').read().splitlines()
            for idx, rel in enumerate(relations):
                add_table(TableISQL(rel), idx)
        else:
            raise ValueError(f'Unknown value {self.config["database"]["engine_class"]} for key database->engine')

        self.columns_total = 0

        for table in self.tables:
            self.columns_total += len(table.idx2column)

        self.size = len(self.table_names)

    def __len__(self):
        if self.size == 0:
            self.size = len(self.table_names)
        return self.size

    def oneHotAll(self):
        return np.zeros((1, self.size))

    def network_size(self):
        return self.TREE_NUM_IN_NET*self.size

