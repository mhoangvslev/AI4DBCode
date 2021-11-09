# Reinforcement Learning with Tree-LSTM for Join Order Selection
Reinforcement Learning with Tree-LSTM for Join Order Selection(RTOS) is an optimizer which focous on Join order Selection(JOS) problem.  RTOS learn from previous queries to build plan for further queries with the help of DQN and TreeLSTM.

# Development

This repository contains a very early version of RTOS.It is written in python and built on PostgreSQL. It now contains the training and testing part of our system. 

It has been fully tested on <a href="http://www.vldb.org/pvldb/vol9/p204-leis.pdf">join order benchmark(JOB)</a>. You can train RTOS and see its' performance on JOB.

This code is still under development and not stable, add issues if you have any questions. We will polish our code and provide further tools. 

# Important parameters
Here we have listed the most important parameters you need to configure to run RTOS on a new database. 

- schemafile
    - <a href ="https://github.com/gregrahn/join-order-benchmark/blob/master/schema.sql"> a sample</a>
- sytheticDir
    - Directory contains the sytheic workload(queries), more sythetic queries will help RTOS make better plans. 
    - It is nesscery when you want apply RTOS for a new database.  
- JOBDir
    - Directory contains all JOB queries. 
- Configure of PostgreSQL
    - dbName : the name of database 
    - userName : the user name of database
    - password : your password of userName
    - ip : ip address of PostgreSQL
    - port : port of PostgreSQL

# Requirements
- Pytorch 1.0
- Python 3.7
- torchfold
- psqlparse

# Run the JOB 
1. Follow the https://github.com/GDD-Nantes/jobrdf to configure the database or download [the dump file](https://jeanbraye.freeboxos.fr:2753/share/zAt-rgPHE6K8nI8S/imdb_pg11) for the database.
2. Add JOB queries in JOBDir.
3. Add sythetic queries in sytheticDir, more sythetic queries will help RTOS generate better results. E.g. You can generate queries by using templates.
4. Cost Training, It will store the model which optimize on cost in PostgreSQL

		python3 CostTraining.py

5. Latency Tuning, It will load the cost training model and generate the latency training model
	
		python3 LatencyTuning.py 

# Docker
An convenient script is available to make things easier:
```bash
sh launch.sh start|build postgres|rtos-cpu|rtos-gpu

# Once inside rtos-cpu  or rtos gpu, launch the commands from previous section
```

1. Create a Postgres container

```bash
sh launch.sh build postgres # Build postgres container
sh launch.sh start postgres # Launch it in background
wget https://jeanbraye.freeboxos.fr:2753/share/zAt-rgPHE6K8nI8S/imdb_pg11 -O imdb_pg11 # Download the dump
sh launch.sh start postgres init path/to/imdb_pg11 #Load the dump into base
```

1. Create a RTOS container

```bash
sh launch.sh build rtos-cpu|gpu # Build rtos container
sh launch.sh start rtos-cpu|gpu # Launch it in background

python CostTraining.py
python LatencyTuning.py
```

# Questions
Contact Xiang Yu (x-yu17@mails.tsinghua.edu.cn) if you have any questions.