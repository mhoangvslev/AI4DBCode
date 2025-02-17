import torch
from torch.functional import Tensor
from torch.nn import init
import torchfold
import torch.nn as nn
import os

import yaml

config = yaml.load(open(os.environ["RTOS_CONFIG"], 'r'), Loader=yaml.FullLoader)[os.environ["RTOS_TRAINTYPE"]]
NB_FEATURE_SLOTS = 2 if config["database"]["engine_class"] == "sql" else 3

class TreeLSTM(nn.Module):
    def __init__(self, num_units):
        super().__init__()
        self.num_units = num_units
        self.FC1 = nn.Linear(num_units, 5 * num_units)
        self.FC2 = nn.Linear(num_units, 5 * num_units)
        self.FC0 = nn.Linear(num_units, 5 * num_units)
        self.LNh = nn.LayerNorm(num_units,)
        self.LNc = nn.LayerNorm(num_units,)
    def forward(self, left_in, right_in,inputX):
        lstm_in = self.FC1(left_in[0])
        lstm_in += self.FC2(right_in[0])
        lstm_in += self.FC0(inputX)
        a, i, f1, f2, o = lstm_in.chunk(5, 1)
        c = (a.tanh() * i.sigmoid() + f1.sigmoid() * left_in[1] +
             f2.sigmoid() * right_in[1])
        h = o.sigmoid() * c.tanh()
        return h,c
class TreeRoot(nn.Module):
    """TreeLSTM's root. The dynamic pooling happens here

    Args:
        nn ([type]): Implements PyTorch's Neural Network module
    """
    def __init__(self,num_units):
        super().__init__()
        self.num_units = num_units
        self.FC = nn.Linear(num_units, num_units)
        self.sum_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        # self.max_pooling = nn.AdaptiveAvgPool2d((1,num_units))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, tree_list):
        return self.relu(self.FC(self.sum_pooling(tree_list)).view(-1,self.num_units))

class SPINN(nn.Module):
    """
    The actual implementation of TreeLSTM. SPINN (https://github.com/stanfordnlp/spinn) combines
    parsing and interpretation within a single tree-sequence hybrid model by integrating tree-structured sentence 
    interpretation into the linear sequential structure of a shift-reduce parser. See https://arxiv.org/abs/1603.06021 

    Args:
        nn ([type]): [description]
    """

    def __init__(self, n_classes, size, n_words, mask_size,device,max_column_in_table = 15):
        super().__init__()
        self.size = size
        self.tree_lstm = TreeLSTM(size)
        self.tree_root = TreeRoot(size)
        self.FC = nn.Linear(size*2, size)
        self.table_embeddings = nn.Embedding(n_words, size)#2 * max_column_in_table * size)
        self.column_embeddings = nn.Embedding(n_words, NB_FEATURE_SLOTS * max_column_in_table * size)

        self.out = nn.Linear(size*2, size)
        self.out2 = nn.Linear(size, n_classes)
        self.outFc = nn.Linear(mask_size, size)
        self.max_pooling = nn.AdaptiveMaxPool2d((1,size))
        self.relu = nn.ReLU()
        self.sigmoid = nn.ReLU()
        self.leafFC = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()
        self.LN1 = nn.LayerNorm(size,)
        self.LN2 = nn.LayerNorm(size,)
        self.max_column_in_table = max_column_in_table
        self.leafLn = nn.LayerNorm(size,)
        self.device = device

    def leaf(self, word_id: torch.Tensor, table_fea: torch.Tensor=None):
        all_columns = table_fea.view(-1,self.max_column_in_table*NB_FEATURE_SLOTS,1)*self.column_embeddings(word_id).reshape(-1,NB_FEATURE_SLOTS * self.max_column_in_table,self.size)
        all_columns = self.relu(self.leafFC(all_columns))
        table_emb = self.max_pooling(all_columns.view(-1,self.max_column_in_table*NB_FEATURE_SLOTS,self.size)).view(-1,self.size)
        return self.leafLn(table_emb), torch.zeros(word_id.size()[0], self.size,device = self.device,dtype = torch.float32)
        
    def inputX(self,left_emb,right_emb):
        cat_emb = torch.cat([left_emb,right_emb],dim = 1)
        return self.relu(self.FC(cat_emb))
    def childrenNode(self, left_h, left_c, right_h, right_c,inputX):
        return self.tree_lstm((left_h, left_c), (right_h, right_c),inputX)
    def root(self,tree_list):
        return self.tree_root(tree_list).view(-1,self.size)
    
    def logits(self, encoding: Tensor, join_matrix: Tensor) -> Tensor :
        encoding = self.root(encoding.view(1,-1,self.size))
        matrix = self.relu(self.outFc(join_matrix))
        outencoding = torch.cat([encoding,matrix],dim = 1)
        return self.out2(self.relu(self.out(outencoding)))