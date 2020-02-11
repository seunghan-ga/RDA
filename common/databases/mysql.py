# -*- coding: utf-8 -*-
import pymysql
import numpy as np
import pandas as pd


class mysql_handler(object):
    def __init__(self):
        self.connection = None

    def connect(self, **info):
        try:
            if self.connection is None:
                self.connection = pymysql.connect(host=info["host"],
                                                  user=info["user"],
                                                  password=info["password"],
                                                  db=info["db"])
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
                    cursor.executemany(query, data)
                else:
                    cursor.execute(query)
                self.connection.commit()
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
        "host": "192.168.1.101",
        "user": "hadoop",
        "password": "hadoop",
        "db": "DEMO"
    }
    connector = mysql_handler()
    res = connector.connect(**db_info)
    print(res)
    data = connector.select("select * from sar order by date limit 10")
    print(data)
    res = connector.close()
    print(res)




