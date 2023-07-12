from mongoengine import connect


class Connection:
    def __init__(self, db='test', host='localhost', port=27017):
        self.db = db
        self.host = host
        self.port = port

    def __enter__(self):
        self.conn = connect(db=self.db, host=self.host, port=self.port)
        return self.conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.close()