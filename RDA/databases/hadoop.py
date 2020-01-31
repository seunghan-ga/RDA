# -*- coding: utf-8 -*-
from impala import dbapi as impyla
import numpy as np
import pandas as pd


class impala_handler(object):
    def __init__(self):
        self.connection = None

    def connect(self, **info):
        try:
            if self.connection is None:
                self.connection = impyla.connect(host=info['host'],
                                                 port=info['port'],
                                                 database=info['db'])
            return 0
        except Exception as e:
            print(e)
            return -1

    def select(self, query):
        try:
            if self.connection is not None:
                cursor = self.connection.cursor()
                cursor.execute(query)
                columns = np.array(cursor.description).T[0]
                data = pd.DataFrame(cursor.fetchall(), columns=columns)
                return data
        except Exception as e:
            print(e)
            return -1

    def insert(self, query, data=None):
        try:
            if self.connection is not None:
                cursor = self.connection.cursor()
                if data is not None:
                    for line in data:
                        query += str(tuple(line)) + ","
                    query = query[:-1]
                cursor.execute(query)
            return 0
        except Exception as e:
            print(e)
            return -1

    def close(self):
        try:
            if self.connection is not None:
                self.connection.close()
            return 0
        except Exception as e:
            print(e)
            return -1


if __name__ == "__main__":
    db_info = {
        "host": "192.168.1.103",
        "port": 21050,
        "db": "demo"
    }

    connector = impala_handler()
    res = connector.connect(**db_info)
    print(res)
    data = connector.select("select * from bearing order by dat limit 10")
    print(data)
    res = connector.close()
    print(res)
