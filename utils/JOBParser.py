import os
import re
import numpy as np
from rdflib.plugins.sparql.operators import Builtin_REGEX, ConditionalAndExpression, ConditionalOrExpression, UnaryMinus, UnaryNot, UnaryPlus
from torch._C import BoolType, Value
from rdflib.term import Literal, Variable, URIRef
from rdflib.plugins.sparql import parserutils

from utils.parser.parsed_query import ParsedQuery
from utils.parser.parser import QueryParser

class Expr:
    def __init__(self, expr, list_kind=0) -> None:
        if os.environ["RTOS_ENGINE"] == "sql":
            self.expr = ExprSQL(expr, list_kind=list_kind)
        elif os.environ["RTOS_ENGINE"] == "sparql":
            self.expr = ExprISQL(expr, list_kind=list_kind)
        else:
            raise ValueError("Unknown value for RTOS_ENGINE")
    
    def isCol(self):
        return self.expr.isCol()
    
    def getValue(self, value_expr):
        return self.expr.getValue(value_expr)

    def getAliasName(self):
        return self.expr.getAliasName()

    def getColumnName(self):
        return self.expr.getColumnName()

    def __str__(self) -> str:
        return self.expr.__str__()

class ExprISQL:
    def __init__(self, expr: parserutils.Expr, list_kind = 0):
        self.expr = expr
        self.list_kind = list_kind
        self.isInt = False
        self.val = 0

    def isCol(self):
        #return isinstance(self.expr, Variable) or isinstance(self.expr, Literal)
        return True

    def getAliasName(self):
        return self.getColumnName()

    def getColumnName(self):
        if isinstance(self.expr, Literal) or isinstance(self.expr, Variable):
            return self.getValue(self.expr)
        elif self.expr._evalfn.__name__ == Builtin_REGEX.__name__ :
            var, _ = self.getValue(self.expr)
            return var
        else:
            raise "No Known type of Expr"

    def getValue(self, value_expr: parserutils.Expr):
        if isinstance(value_expr, Literal):
            if value_expr.datatype == URIRef('http://www.w3.org/2001/XMLSchema#string'):
                return value_expr.value
            elif value_expr.datatype == URIRef('http://www.w3.org/2001/XMLSchema#integer'):
                self.isInt = True
                self.val = value_expr.value
                return str(self.val)
            else:
                return rf"'{str(value_expr)}'"
        elif isinstance(value_expr, Variable):
            return f'?{str(value_expr)}'

        elif value_expr._evalfn.__name__ == Builtin_REGEX.__name__:
            return self.getValue(value_expr.get('text')), self.getValue(value_expr.get('pattern'))
        else:
            raise ValueError("Unknown value in Expr")

    def __str__(self,):
        if isinstance(self.expr, Literal) or isinstance(self.expr, Variable):
            return self.getValue(self.expr)
        elif self.expr._evalfn.__name__ == Builtin_REGEX.__name__ :
            var, pattern = self.getValue(self.expr)
            return f'regex({var}, {pattern})'
        else:
            raise "No Known type of Expr"

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

class TargetTable:
    def __init__(self, target, alias=None) -> None:
        if os.environ["RTOS_ENGINE"] == "sql":
            self.target = TargetTableSQL(target)
        elif os.environ["RTOS_ENGINE"] == "sparql":
            self.target = TargetTableISQL(target, alias)
        else:
            raise ValueError("Unknown RTOS_ENGINE value")
    
    def getValue(self):
        return self.target.getValue()

    def __str__(self) -> str:
        return self.target.__str__()
        
class TargetTableSQL:
    def __init__(self, target):
        """
        {'location': 7, 'name': 'alternative_name', 'val': {'FuncCall': {'funcname': [{'String': {'str': 'min'}}], 'args': [{'ColumnRef': {'fields': [{'String': {'str': 'an'}}, {'String': {'str': 'name'}}], 'location': 11}}], 'location': 7}}}
        """
        self.target = target
    #         print(self.target)

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

class TargetTableISQL:
    def __init__(self, target, alias=None) -> None:
        self.target = target
        self.alias = alias

    def getAliasName(self):
        return self.alias
    
    def getFullName(self):
        return self.target

    def __str__(self):
        return f'{self.getAliasName()}. \n'

    def getValue(self):
        return self.target


class FromTable:
    def __init__(self, from_table, alias=None):
        if os.environ["RTOS_ENGINE"] == "sql":
            self.from_table = FromTableSQL(from_table)
        elif os.environ["RTOS_ENGINE"] == "sparql":
            self.from_table = FromTableISQL(from_table, alias)
        else:
            raise ValueError("Unknown RTOS_ENGINE value")

    def getFullName(self):
        return self.from_table.getFullName()

    def getAliasName(self):
        return self.from_table.getAliasName()
    
    def __str__(self) -> str:
        return self.from_table.__str__()


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

class FromTableISQL:
    def __init__(self, from_table, alias):
        """
        from_table = predicate
        alias = s p o
        """
        self.from_table = from_table
        self.alias = alias

    def getFullName(self):
        return self.from_table

    def getAliasName(self, debug = False):
        return self.alias

    def __str__(self):
        return f'{self.getAliasName()} .\n' 

class Comparison:
    def __init__(self, comparison) -> None:
        if os.environ["RTOS_ENGINE"] == "sql":
            self.comparision = ComparisonSQL(comparison)
        elif os.environ["RTOS_ENGINE"] == "sparql":
            self.comparision = ComparisionISQL(comparison)
        else:
            raise ValueError("Unknown value RTOS_ENGINE...")

    def isCol(self):
        return self.comparision.isCol()

    def __str__(self):
        return self.comparision.__str__()

class ComparisionISQL:
    def __init__(self, comparison) -> None:

        def get_kind(comp: parserutils.Expr):
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
        self.comp_list = []
        self.aliasname_list = []
        self.kind = None
        self.lexpr = None
        self.rexpr = None

        #print(self.comparison, type(self.comparison))

        if "Relational" in self.comparison.name:
            self.lexpr = Expr(self.comparison.get("expr"))
            self.kind = self.comparison.get("op")
            if not isinstance(self.comparison.get("other"), parserutils.Expr):
                self.rexpr = Expr(self.comparison.get("other"),self.kind)
            else:
                self.rexpr = ComparisionISQL(self.comparison.get("other"))

            self.aliasname_list = []

            # self.aliasname_list.append(str(self))
            # self.column_list.append(str(self))

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0

        elif "Unary" in self.comparison.name:
            self.lexpr = Expr(self.comparison.get('expr'))
            self.kind = get_kind(self.comparison)

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

        elif "Conditional" in self.comparison.name:
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

            self.kind = get_kind(self.comparison)
            self.comp_list.append(ComparisionISQL(self.comparison.get('expr')))
            self.comp_list.extend([ComparisionISQL(x) for x in self.comparison.get('other')])

            self.aliasname_list = []
            for comp in self.comp_list:
                if comp.lexpr.isCol():
                    self.lexpr = comp.lexpr
                    # self.aliasname_list.append(str(comp))
                    # self.column_list.append(str(comp))

                    self.aliasname_list.append(comp.lexpr.getAliasName())
                    self.column_list.append(comp.lexpr.getColumnName())

                    #break

            self.comp_kind = 2
        else:
            raise NotImplementedError(f"No handler for {self.comparison.name}")

    def isCol(self):
        return False
    
    def __str__(self, final=True):
        res = ""
        if len(self.comp_list) == 0:
            if "Relational" in self.comparison.name or "Conditional" in self.comparison.name:
                res += f'{ str(self.lexpr) } { self.kind } { str(self.rexpr) }'
            elif "Unary" in self.comparison.name:
                res += f'{ self.kind }{ str(self.lexpr) }'
        else:
            res += f' {self.kind} '.join([ comp.__str__(final=False) for comp in self.comp_list ])
        
        return f'FILTER({ res })' if final else res

class ComparisonSQL:
    def __init__(self, comparison):
        self.comparison = comparison
        self.column_list = []
        if "A_Expr" in self.comparison:
            self.lexpr = Expr(comparison["A_Expr"]["lexpr"])
            self.kind = comparison["A_Expr"]["kind"]
            if not "A_Expr" in comparison["A_Expr"]["rexpr"]:
                self.rexpr = Expr(comparison["A_Expr"]["rexpr"],self.kind)
            else:
                self.rexpr = Comparison(comparison["A_Expr"]["rexpr"])

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())

            if self.rexpr.isCol():
                self.aliasname_list.append(self.rexpr.getAliasName())
                self.column_list.append(self.rexpr.getColumnName())

            self.comp_kind = 0
        elif "NullTest" in self.comparison:
            self.lexpr = Expr(comparison["NullTest"]["arg"])
            self.kind = comparison["NullTest"]["nulltesttype"]

            self.aliasname_list = []

            if self.lexpr.isCol():
                self.aliasname_list.append(self.lexpr.getAliasName())
                self.column_list.append(self.lexpr.getColumnName())
            self.comp_kind = 1
        else:
            #             "boolop"
            self.kind = comparison["BoolExpr"]["boolop"]
            self.comp_list = [Comparison(x)
                              for x in comparison["BoolExpr"]["args"]]
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
                print(json.dumps(self.comparison, sort_keys=True, indent=4))
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

class Table:
    def __init__(self, table_tree=None, table_name=None) -> None:
        if os.environ["RTOS_ENGINE"] == "sql":
            self.table = TableSQL(table_tree)
        if os.environ["RTOS_ENGINE"] == "sparql":
            self.table = TableISQL(table_name)
    
    def oneHotAll(self):
        return self.table.oneHotAll()

class TableSQL:
    def __init__(self, table_tree):
        self.name = table_tree["relation"]["RangeVar"]["relname"]
        self.column2idx = {}
        self.idx2column = {}
        for idx, columndef in enumerate(table_tree["tableElts"]):
            self.column2idx[columndef["ColumnDef"]["colname"]] = idx
            self.idx2column[idx] = columndef["ColumnDef"]["colname"]

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))

class TableISQL:
    def __init__(self, table_name) -> None:
        self.name = table_name
        self.column2idx = {}
        self.idx2column = { v:k for k, v in self.column2idx.items() }

    def updateTable(self, newCol):
        #print(self.name, self.column2idx)
        if newCol not in self.column2idx.keys():
            idx = len(self.column2idx)
            self.column2idx[newCol] = idx
            self.idx2column[idx] = newCol

    def oneHotAll(self):
        return np.zeros((1, len(self.column2idx)))
        

class DB:
    def __init__(self, schema,TREE_NUM_IN_NET=40):

        from psqlparse import parse_dict
        parse_tree = parse_dict(schema)

        self.tables = {}
        self.name2idx = {}
        self.table_names = []
        self.name2table = {}
        self.size = 0
        self.TREE_NUM_IN_NET = TREE_NUM_IN_NET

        def add_table(table, idx):
            self.tables[table.name] = table
            self.table_names.append(table.name)
            self.name2idx[table.name] = idx
            self.name2table[table.name] = self.tables[table.name]

        if os.environ["RTOS_ENGINE"] == "sql":
            for idx, table_tree in enumerate(parse_tree):
                table_name = table_tree["CreateStmt"]["relation"]["RangeVar"]["relname"]
                add_table(Table(table_tree["CreateStmt"], table_name=table_name), idx)

        elif os.environ["RTOS_ENGINE"] == "sparql":
            for i, table_tree in enumerate(parse_tree):
                for j, columndef in enumerate(table_tree["CreateStmt"]["tableElts"]):
                    table_name = table_tree["CreateStmt"]["relation"]["RangeVar"]["relname"]
                    if table_name == "title":
                        table_name = "title_t"
                    
                    column_name = columndef["ColumnDef"]["colname"]
                    table_name = f'http://imdb.org/{table_name}#{column_name}'
                    add_table(TableISQL(table_name), i+j)
            
            for shortcut in [
                "http://imdb.org/movie_companies#movie_info_idx#movie_id#movie_id"
            ]:
                add_table(TableISQL(shortcut), len(self.table_names))
        else:
            raise ValueError("Unknown value for env variable RTOS_ENGINE")

        self.columns_total = 0

        for table in self.tables.values():
            self.columns_total += len(table.idx2column)

        self.size = len(self.table_names)

    def __len__(self,):
        if self.size == 0:
            self.size = len(self.table_names)
        return self.size

    def oneHotAll(self,):
        return np.zeros((1, self.size))

    def network_size(self,):
        return self.TREE_NUM_IN_NET*self.size

