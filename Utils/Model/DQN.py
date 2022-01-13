
import logging
import math
import os
import random
import time
from typing import Any, List, Tuple, Union
from torch._C import device
from torch.functional import Tensor
from torch.types import Number
import torchfold
import torch.nn.functional as F
#import torchvision.transforms as T
import torch
from collections import namedtuple
from torchfold.torchfold import Fold

from tqdm import tqdm
from Utils.DB.DBUtils import DBRunner
from Utils.Parser.JOBParser import DB
from Utils.Model.TreeLSTM import SPINN
from Utils.DB.QueryUtils import JoinTree, Query
import torch.optim as optim
import numpy as np
from math import log
from itertools import count
import pandas as pd
from scipy.stats import gmean
 
FOOP_CONST=10e13

class ENV(object):
    def __init__(self, sql: Query, db_info: DB, pgrunner: DBRunner, device: device, config: dict):
        self.config = config
        self.sel = JoinTree(sql,db_info,pgrunner,device)
        self.sql = sql
        self.hashs = ""
        self.table_set = set([])
        self.res_table = []
        self.init_table = None
        self.planSpace = int(config["database"]["use_bushy_tree"]) #0:leftDeep,1:bushy
        self.terminate = False

        self._rewarder = config["model"]["rewarder"]
        logging.debug(f"Rewarder: {self._rewarder}")


    def getPlan(self):
        return self.sel.plan

    def actionValue(self, left: str, right: str, model: SPINN) -> Tensor:
        self.sel.joinTables(left,right,fake = True)
        res_Value = self.selectValue(model)
        self.sel.total -= 1
        self.sel.aliasnames_root_set.remove(self.sel.total)
        self.sel.aliasnames_fa.pop(self.sel.left_son[self.sel.total])
        self.sel.aliasnames_fa.pop(self.sel.right_son[self.sel.total])
        return res_Value

    @property
    def rewarder(self):
        return self._rewarder

    @rewarder.setter
    def rewarder(self, rewarder_type: str):
        self._rewarder = rewarder_type

    def selectValue(self, model: SPINN) -> Tensor:
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
                tree_state.append(self.sel.encode_tree_regular(model,idx))
        res = torch.cat(tree_state,dim = 0)
        return model.logits(res, self.sel.join_matrix)

    def selectValueFold(self,fold: Fold):
        tree_state = []
        for idx in self.sel.aliasnames_root_set:
            if not idx in self.sel.aliasnames_fa:
                tree_state.append(self.sel.encode_tree_fold(fold,idx))
            #         res = torch.cat(tree_state,dim = 0)
        return tree_state
        return fold.add('logits',tree_state,self.sel.join_matrix)

    def takeAction(self,left,right):
        self.sel.joinTables(left,right)
        self.hashs += left
        self.hashs += right
        self.hashs += " "

    def hashcode(self):
        return self.sql.sql+self.hashs

    def allAction(self, model: SPINN) -> Tensor:
        """List all possible action for the model

        Args:
            model ([type]): [description]

        Returns:
            List[Any]: List of all possible actions
        """
        action_value_list = []

        for one_join in self.sel.join_candidate:
            
            l_node, r_node = one_join[0], one_join[1]

            l_fa = self.sel.findFather(l_node)
            r_fa = self.sel.findFather(r_node)
            if self.planSpace == 0:
                """If left linear join tree, check if right parent is same as right node and left parent is not left node
                """
                flag1 = ( r_node == r_fa and l_fa != l_node )
                if l_fa != r_fa and (self.sel.total == 0 or flag1):
                    action_value_list.append((self.actionValue(l_node,r_node,model),one_join))
            
            elif self.planSpace == 1:
                if l_fa != r_fa:
                    action_value_list.append((self.actionValue(l_node,r_node,model),one_join))

        logging.debug(f"Possible actions: {len(action_value_list)} out of {len(self.sel.join_candidate)} join candidates")
        return action_value_list

    def reward(self, forceLatency=False):
        """Calculate the reward for the learner. The network seeks to minimise the loss
        (1) rtos: 
            max(cost/baseline_cost - 1, 0)
        (2) Cost improvement: 
            min(
                max(
                    log(cost / baseline_cost, 10), 
                    -10
                ), 
                10
            )
        (3) Cost: cost

        Returns:
            [type]: [description]
        """
        table_list = self.sel.all_table_list if self.config["database"]["engine_class"] == "sparql" else self.sel.from_table_list
        total = self.sel.total + 1
        logging.debug(f"Selected total: {total} out of {len(table_list)}")

        if total == len(table_list):           
            prediction, cost = self.sel.plan2Cost(forceLatency=forceLatency)
            baseline_cost = self.sql.getDPlatency(forceLatency=forceLatency)
            reward = 0

            if self._rewarder == "rtos":
                reward = max(cost/baseline_cost - 1, 0)
            elif self._rewarder == "cost-improvement":
                reward = min(max(np.log10(cost / baseline_cost), -10), 10)
            elif self._rewarder == "cost":
                reward = cost
            elif self._rewarder == "foop-cost":
                reward = 10 * np.sqrt(cost/FOOP_CONST) if cost < FOOP_CONST else 10
            elif self._rewarder == "refined-cost-improvement":
                reward = min(max(np.log10(cost / baseline_cost), -10), 10)
            else:
                raise NotImplementedError(f"No handler for rewarder of type {self._rewarder}!")

            return prediction, cost, reward, True
        else:
            return None, None, 0, False



Transition = namedtuple('Transition',
                        ('env', 'next_value', 'this_value'))
# bestJoinTreeValue = {}
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory: List[Transition] = []
        self.position = 0
        self.bestJoinTreeValue = {}
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        data =  Transition(*args)
        hashv = data.env.hashcode()
        next_value = data.next_value
        if hashv in self.bestJoinTreeValue and self.bestJoinTreeValue[hashv]<data.this_value:
            if self.bestJoinTreeValue[hashv]<next_value:
                next_value = self.bestJoinTreeValue[hashv]
        else:
            self.bestJoinTreeValue[hashv]  = data.this_value
        data = Transition(data.env,self.bestJoinTreeValue[hashv],data.this_value)
        position = self.position
        self.memory[position] = data
        #         self.position
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> List[Transition]:
        if len(self.memory)>batch_size:
            return random.sample(self.memory, batch_size)
        else:
            return self.memory

    def __len__(self):
        return len(self.memory)
    def resetMemory(self,):
        self.memory =[]
    def resetbest(self):
        self.bestJoinTreeValue = {}

class DQN:
    def __init__(self,policy_net: SPINN, target_net: SPINN, db_info: DB, pgrunner: DBRunner, device: device, config: dict):
        self.config = config
        self.Memory = ReplayMemory(1000)
        self.BATCH_SIZE = 1

        self.optimizer = optim.Adam(policy_net.parameters(),lr=3e-4,betas=(0.9,0.999))

        self.steps_done = 0
        self.max_action = 25
        self.EPS_START = 0.4
        self.EPS_END = 0.2
        self.EPS_DECAY = 400
        self.policy_net = policy_net
        self.target_net = target_net
        self.db_info = db_info
        self.pgrunner = pgrunner
        self.device = device
        self.steps_done = 0

    def select_action(self, env: ENV, need_random = True) -> Tuple[Tensor, Tuple[str, str], Tensor]:
        """Decide the next action. During earlier episodes, actions will be chosen randomly but as the training goes on, only actions with minimal costs are chosen

        Args:
            env (ENV): [description]
            need_random (bool, optional): [description]. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: tuple of possible actions, chosen action, all actions
        """

        sample = random.random()
        if need_random:
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                      math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
        else:
            eps_threshold = -1

        action_list = env.allAction(self.policy_net)
        action_batch = torch.cat([x[0] for x in action_list],dim = 1)

        if sample > eps_threshold:
            return (
                action_batch, 
                action_list[torch.argmin(action_batch,dim = 1)[0]][1],
                [x[1] for x in action_list]
            )
        else:
            return (
                action_batch,
                action_list[random.randint(0,len(action_list)-1)][1],
                [x[1] for x in action_list]
            )


    def validate(self, val_list: List[Query], tryTimes = 1, forceLatency=False, infos=dict()) -> Union[str, pd.DataFrame]:
        """[summary]

        MRC: Mean Relevant Cost. MRC=1 means the model's cost is same as PG.
        (G)MRL: (Geometric) Mean Relevant Latency, applied to the ratio between model's latency and db engine. 
            Given that the latency scores for each query follow a certain probability distribution,
            the geometric mean (n-th root of product) indicates the most frequent value (a typical value) of that distribution.
            In another word, when GMRL approaches 0, the model is worse than Postgres and when it's n > 1, it means that the model 
            performs n times better than Postgres.

            e.g: GMRL=2 means that overall, the model is twice better than PG 

        Args:
            val_list ([type]): [description]
            tryTimes (int, optional): [description]. Defaults to 1.

        Returns:
            [type]: [description]
        """
        rewards = []
        prt = []
        mes = 0
        result = None
        prediction = None

        for sql in val_list:
            pg_cost = sql.getDPlatency(forceLatency=forceLatency)
            env = ENV(sql,self.db_info,self.pgrunner,self.device, self.config)

            for t in count():
                action_list, chosen_action, all_action = self.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                prediction, cost, reward, done = env.reward(forceLatency=forceLatency)
                
                if done:
                    rewards.append(reward)
                    mes += reward

                    data = {
                            "query": sql.filename,
                            "step": t,
                            "reward": reward,
                            "cost": cost,
                            "base-cost": pg_cost
                    }
                    data.update(infos)

                    if result is None:
                        result = pd.DataFrame(data, index=[0])
                    else:
                        result = result.append(data, ignore_index=True)

                    break

        mrc, gmrl = np.average(rewards), gmean(rewards)     

        logging.debug(f"MRC: {mrc}n GMRL: {gmrl}")
        return prediction, result

    def optimize_model(self) -> Tuple[Number, float, float]:
        startTime = time.time()
        samples = self.Memory.sample(64)
        value_now_list = []
        next_value_list = []
        if (len(samples)==0):
            return
        
        fold = torchfold.Fold(cuda=(str(self.device) == "cuda"))
        nowL = []
        for one_sample in samples:
            nowList = one_sample.env.selectValueFold(fold)
            nowL.append(len(nowList))
            value_now_list+=nowList
        res = fold.apply(self.policy_net, [value_now_list])[0]
        total = 0
        value_now_list = []
        next_value_list = []
        for idx,one_sample in enumerate(samples):
            value_now_list.append(self.policy_net.logits(res[total:total+nowL[idx]] , one_sample.env.sel.join_matrix ))
            next_value_list.append(one_sample.next_value)
            total += nowL[idx]
        value_now = torch.cat(value_now_list,dim = 0)
        next_value = torch.cat(next_value_list,dim = 0)
        endTime = time.time()

        pre_descend_time = endTime - startTime
        loss = F.smooth_l1_loss(value_now,next_value,size_average=True)
        self.optimizer.zero_grad()

        delta_start = time.time()
        loss.backward()
        delta_end = time.time()
        delta_gd_time = delta_end - delta_start

        self.optimizer.step()
        return loss.item(), pre_descend_time * 1e3, delta_gd_time * 1e3
