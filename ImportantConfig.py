import os

class Config:
    def __init__(self):
        self.sytheticDir = "Queries/sytheic"
        self.JOBDir = os.environ["RTOS_JOB_DIR"]
        self.schemaFile = os.environ["RTOS_SCHEMA_FILE"]
        self.dbName = os.environ["RTOS_DB_NAME"]
        self.userName = os.environ["RTOS_DB_USER"]
        self.password = os.environ["RTOS_DB_PASSWORD"]
        self.ip = os.environ["RTOS_DB_HOST"]
        self.port = os.environ["RTOS_DB_PORT"]
