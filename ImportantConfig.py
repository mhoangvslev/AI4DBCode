import os

class Config:
    def __init__(self):
        # Postgres
        self.sytheticDir = "Queries/sytheic"
        self.JOBDir = os.environ["RTOS_JOB_DIR"]
        self.schemaFile = os.environ["RTOS_SCHEMA_FILE"]
        self.sql_dbName = os.environ["RTOS_DB_NAME"]
        self.sql_userName = os.environ["RTOS_DB_USER"]
        self.sql_password = os.environ["RTOS_DB_PASSWORD"]
        self.sql_ip = os.environ["RTOS_DB_HOST"]
        self.sql_port = os.environ["RTOS_DB_PORT"]

        # Virtuoso
        self.isql_endpoint = os.environ["RTOS_ISQL_ENDPOINT"]
        self.isql_graph = os.environ["RTOS_ISQL_GRAPH"]
        self.isql_host = os.environ["RTOS_ISQL_HOST"]
        self.isql_port = os.environ["RTOS_ISQL_PORT"]
