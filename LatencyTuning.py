
from datetime import datetime
import logging
from typing import AnyStr, List, Tuple
from utils.DBUtils import PGRunner, ISQLRunner
from utils.sqlSample import sqlInfo
import numpy as np
from itertools import count
from math import log
import random
import time
from DQN import DQN,ENV
from utils.TreeLSTM import SPINN
from utils.JOBParser import DB
import copy
import torch
from torch.nn import init
from ImportantConfig import Config

import os
import argparse
import pandas as pd
from glob import glob
from tqdm import tqdm

class LatencyTuning:
    def __init__(self, rewarder) -> None:
        self.handlers = [
            logging.FileHandler(f"cost-training_{datetime.now()}.log"),
            #logging.StreamHandler()
        ]

        logging.basicConfig(
            level="DEBUG",
            handlers=self.handlers,
            format='%(asctime)s - %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S'
        )

        self.rewarder = rewarder

        self.config = Config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if os.environ['RTOS_PYTORCH_DEVICE'] == "gpu" else torch.device("cpu")
        #device = torch.device("cpu")

        with open(self.config.schemaFile, "r") as f:
            createSchema = "".join(f.readlines())

        self.db_info = DB(createSchema)

        self.featureSize = 128

        self.policy_net = SPINN(
            n_classes = 1, size = self.featureSize, 
            n_words = self.config.n_words,
            mask_size= len(self.db_info)*len(self.db_info),
            device=self.device, 
            max_column_in_table=self.config.max_column_in_table
        ).to(self.device)

        self.target_net = SPINN(
            n_classes = 1, size = self.featureSize, 
            n_words = self.config.n_words, 
            mask_size= len(self.db_info)*len(self.db_info),
            device=self.device, 
            max_column_in_table=self.config.max_column_in_table
        ).to(self.device)

        for name, param in self.policy_net.named_parameters():
            logging.debug(f"Parameter: {name} of shape {param.shape}")
            if len(param.shape)==2:
                init.xavier_normal(param)
            else:
                init.uniform(param)

        # policy_net.load_state_dict(torch.load("models/JOB_tc.pth"))#load cost train model
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


        self.runner = (
            PGRunner(
                self.config.sql_dbName,
                self.config.sql_userName,
                self.config.sql_password,
                self.config.sql_ip,
                self.config.sql_port,
                isCostTraining=True,
                latencyRecord = False,
                latencyRecordFile = "Cost.json"
            ) if os.environ["RTOS_ENGINE"] == "sql" else
            ISQLRunner(
                self.config.isql_endpoint,
                self.config.isql_graph,
                self.config.isql_host,
                self.config.isql_port,
                isCostTraining=True,
                latencyRecord = False,
                latencyRecordFile = "Cost.json"
            )
        )

        self.dqn = DQN(self.policy_net,self.target_net,self.db_info,self.runner, self.device, self.rewarder)

    def k_fold(self, input_list: List[sqlInfo],k,ix = 0) -> Tuple[List[sqlInfo], List[sqlInfo]]:
        li = len(input_list)
        kl = (li-1)//k + 1
        train = []
        validate = []
        for idx in range(li):

            if idx%k == ix:
                validate.append(input_list[idx])
            else:
                train.append(input_list[idx])
        return train, validate


    def QueryLoader(self, QueryDir: str) -> List[sqlInfo]:
        def file_name(file_dir):
            import os
            L = []
            for root, dirs, files in os.walk(file_dir):
                for file in files:
                    if os.path.splitext(file)[1] == f'.{os.environ["RTOS_ENGINE"]}':
                        L.append(os.path.join(root, file))
            return L
        files = file_name(QueryDir)
        sql_list = []
        for filename in files:
            with open(filename, "r") as f:
                data = f.readlines()
                one_sql = "".join(data)
                sql_list.append(sqlInfo(self.runner,one_sql,filename))
        return sql_list

    def resample_sql(self, sql_list: List[sqlInfo]):
        rewards = []
        reward_sum = 0
        rewardsP = []
        mes = 0
        for sql in sql_list:
            #         sql = val_list[i_episode%len(train_list)]
            pg_cost = sql.getDPlatency()
            #         continue
            env = ENV(sql,self.db_info,self.runner,self.device, rewarder=rewarder)

            for t in count():
                action_list, chosen_action, all_action = self.dqn.select_action(env,need_random=False)

                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)

                reward, done = env.reward()
                if done:
                    # mrc = max(np.exp(reward*log(1.5))/pg_cost-1,0)
                    # rewardsP.append(np.exp(reward*log(1.5)-log(pg_cost)))
                    # mes += reward*log(1.5)-log(pg_cost)

                    mrc = reward
                    rewardsP.append(mrc)
                    mes += mrc
                    rewards.append((mrc,sql))
                    reward_sum += mrc
                    break
        import random
        logging.debug(rewardsP)
        res_sql = []
        logging.debug(mes/len(sql_list))
        for idx in range(len(sql_list)):
            rd = random.random()*reward_sum
            for ts in range(len(sql_list)):
                rd -= rewards[ts][0]
                if rd<0:
                    res_sql.append(rewards[ts][1])
                    break
        return res_sql+sql_list

    def predict(self, queryfiles: List[AnyStr]) -> str:

        for queryfile in queryfiles:
            logging.debug(f"Processing {queryfile}...")
            sqlt = sqlInfo(self.runner, open(queryfile, "r").read(), queryfile)
            env = ENV(sqlt,self.db_info,self.runner,self.device, rewarder=rewarder)

            previous_state_list = []
            action_this_epi = []
            nr = True
            nr = random.random()>0.3 or sqlt.getBestOrder()==None
            acBest = (not nr) and random.random()>0.7
            for t in count():
                # beginTime = time.time();
                action_list, chosen_action,all_action = self.dqn.select_action(env,need_random=nr)
                value_now = env.selectValue(self.policy_net)
                next_value = torch.min(action_list).detach()
                # e1Time = time.time()
                env_now = copy.deepcopy(env)
                # endTime = time.time()
                # logging.debug(f"make {endTime-startTime,endTime-e1Time}")
                if acBest:
                    chosen_action = sqlt.getBestOrder()[t]
                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)
                action_this_epi.append((left,right))

                reward, done = env.reward()
                reward = torch.tensor([reward], device=self.device, dtype = torch.float32).view(-1,1)

                previous_state_list.append((value_now,next_value.view(-1,1),env_now))
                if done:
                    next_value = 0
                    sqlt.updateBestOrder(reward.item(),action_this_epi)

                expected_state_action_values = (next_value ) + reward.detach()
                final_state_value = (next_value ) + reward.detach()

                if done:
                    cnt = 0
                    #             for idx in range(t-cnt+1):
                    global tree_lstm_memory
                    tree_lstm_memory = {}
                    self.dqn.Memory.push(env,expected_state_action_values,final_state_value)
                    for pair_s_v in previous_state_list[:0:-1]:
                        cnt += 1
                        if expected_state_action_values > pair_s_v[1]:
                            expected_state_action_values = pair_s_v[1]
                        #                 for idx in range(cnt):
                        expected_state_action_values = expected_state_action_values
                        self.dqn.Memory.push(pair_s_v[2],expected_state_action_values,final_state_value)
                    #                 break
                    
                    print("================================")
                    print(f"Prediction for { queryfile }:")
                    print(env.getPlan())
                    print(f"Reward: {reward}")
                    print("================================")
                    break

    def train(self, trainSet,validateSet):

        trainSet_temp = trainSet
        losses = []
        startTime = time.time()
        print_every = 20
        TARGET_UPDATE = 3
        for i_episode in tqdm(range(0,10000)):
            if i_episode % 200 == 100:
                trainSet = self.resample_sql(trainSet_temp)
            #     sql = random.sample(train_list_back,1)[0][0]
            sqlt = random.sample(trainSet[0:],1)[0]
            pg_cost = sqlt.getDPlatency()
            env = ENV(sqlt,self.db_info,self.runner,self.device, rewarder=rewarder)

            previous_state_list = []
            action_this_epi = []
            nr = True
            nr = random.random()>0.3 or sqlt.getBestOrder()==None
            acBest = (not nr) and random.random()>0.7
            for t in count():
                # beginTime = time.time();
                action_list, chosen_action,all_action = self.dqn.select_action(env,need_random=nr)
                value_now = env.selectValue(self.policy_net)
                next_value = torch.min(action_list).detach()
                # e1Time = time.time()
                env_now = copy.deepcopy(env)
                # endTime = time.time()
                # logging.debug(f"make {endTime-startTime,endTime-e1Time}")
                if acBest:
                    chosen_action = sqlt.getBestOrder()[t]
                left = chosen_action[0]
                right = chosen_action[1]
                env.takeAction(left,right)
                action_this_epi.append((left,right))

                reward, done = env.reward()
                reward = torch.tensor([reward], device=self.device, dtype = torch.float32).view(-1,1)

                previous_state_list.append((value_now,next_value.view(-1,1),env_now))
                if done:

                    #             logging.debug("done")
                    next_value = 0
                    sqlt.updateBestOrder(reward.item(),action_this_epi)

                expected_state_action_values = (next_value ) + reward.detach()
                final_state_value = (next_value ) + reward.detach()

                if done:
                    cnt = 0
                    #             for idx in range(t-cnt+1):
                    global tree_lstm_memory
                    tree_lstm_memory = {}
                    self.dqn.Memory.push(env,expected_state_action_values,final_state_value)
                    for pair_s_v in previous_state_list[:0:-1]:
                        cnt += 1
                        if expected_state_action_values > pair_s_v[1]:
                            expected_state_action_values = pair_s_v[1]
                        #                 for idx in range(cnt):
                        expected_state_action_values = expected_state_action_values
                        self.dqn.Memory.push(pair_s_v[2],expected_state_action_values,final_state_value)
                    #                 break
                    loss = 0

                if done:
                    # break
                    loss = self.dqn.optimize_model()
                    loss = self.dqn.optimize_model()
                    loss = self.dqn.optimize_model()
                    loss = self.dqn.optimize_model()
                    losses.append(loss)
                    if ((i_episode + 1)%print_every==0):
                        logging.debug(np.mean(losses))
                        logging.debug(f"###################### Epoch {i_episode//print_every}, baseline_cost = {pg_cost}")

                        mrc, gmrl = self.dqn.validate(validateSet)
                        training_time = time.time()-startTime

                        fn = os.path.join(Config().JOBDir, "validation.csv")
                        pd.DataFrame(
                            [[i_episode+1, training_time, mrc, gmrl, pg_cost]], 
                            columns=["episode", "training_time", "mrc", "gmrl", "pg_cost"]
                        ).to_csv(fn, mode="a", header=(not os.path.exists(fn)), index=False)
                        logging.debug(f"time {training_time}")
                        logging.debug("~~~~~~~~~~~~~~")
                    break
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        torch.save(self.policy_net.cpu().state_dict(), 'models/LatencyTuning.pth')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", help='Mode: train | predict')
    parser.add_argument('--queryfile', type=str, default="", nargs="*", help="Relative path to queryfile")
    parser.add_argument('--log_level', type=str, default="DEBUG", help="Log level")
    parser.add_argument('--reward', type=str, default="rtos", help="The type of reward rtos|cost-improvement|cost|foop-cost")
    args = parser.parse_args()

    rewarder = args.reward

    lt = LatencyTuning(rewarder=rewarder)

    if args.mode == "train":
        sytheticQueries = lt.QueryLoader(QueryDir=lt.config.sytheticDir)
        # logging.debug(sytheticQueries)
        JOBQueries = lt.QueryLoader(QueryDir=lt.config.JOBDir)
        Q4,Q1 = lt.k_fold(JOBQueries,10,1)
        # logging.debug(Q4,Q1)
        lt.train(Q4+sytheticQueries,Q1)
    elif args.mode == "predict":
        queryfiles: List[AnyStr] = []
        for q in args.queryfile:
            queryfiles.extend(glob(q))

        lt.predict(queryfiles)
