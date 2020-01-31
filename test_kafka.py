from __future__ import absolute_import, division, print_function, unicode_literals

from confluent_kafka import KafkaException
from confluent_kafka import TopicPartition
from confluent_kafka import Consumer
import pymysql
import datetime
import time
import sys


def receive_data(consumer, windowsize=30, topic=None, partition=None, offset=None):
    try:
        data = []
        close_time = time.time() + windowsize

        if partition is None and offset is None:
            pass
        else:
            consumer.seek(TopicPartition(topic, partition, offset))

        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if time.time() > close_time:
                offset = msg.offset()
                partition = msg.partition()
                break

            if msg.error():
                # raise KafkaException(msg.error()) # test
                continue
            else:
                data.append(msg.value().decode('utf-8').split(','))

        return partition, offset, data

    except KeyboardInterrupt:
        sys.stderr.write('%% Aborted by user\n')
        return None


if __name__ == "__main__":
    conf = {'bootstrap.servers': "192.168.1.101:9092",
            'group.id': "test",
            'session.timeout.ms': 6000,
            'auto.offset.reset': 'earliest'}
    topic = 'test'
    windowsize = 30
    partition = None
    offset = None
    consumer = Consumer(conf)
    consumer.subscribe([topic])

    db_info = {
        "host": "192.168.1.101",
        "user": "hadoop",
        "password": "hadoop",
        "db": "DEMO",
        "charset": "utf8"
    }
    # conn = pymysql.connect(**db_info)

    try:
        while True:
            partition, offset, data = receive_data(consumer, windowsize, topic, partition, offset)
            print("partition :", partition)
            print("offset :", offset)
            print("length :", len(data), " data: ", data)

            now = datetime.datetime.now()
            now_str = datetime.datetime.strftime(now, "%Y-%m-%d ")

            insert_data = []
            for line in data:
                split = line[0].replace("    ", " ").split()
                split[0] = now_str + split[0]
                insert_data.append(split)

            print(insert_data)

            # cursor = conn.cursor()
            # sql = "insert into sar values(%s, %s, %s, %s, %s, %s, %s, %s)"
            # cursor.executemany(sql, insert_data)
            # conn.commit()

    except Exception as e:
        print(e)

    finally:
        # conn.close()
        consumer.close()
