class Credentials( object ):
    def __init__(self):
        self.host = "comunidade-ds-postgres.c50pcakiuwi3.us-east-1.rds.amazonaws.com"
        self.database = "comunidadedsdb"
        self.port = "5432"
        self.username = "member"
        self.password = "cdspa"
        
    def database_connection(self):
        self.database = f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return(self.database)