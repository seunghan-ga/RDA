# -*- coding: utf-8 -*-
from RDA.analysis.thrsehold import iqr_threshold
from RDA.databases import hadoop
import pandas as pd
import numpy as np


if __name__ == "__main__":
    db_info = {
        "host": "192.168.1.103",
        "port": 21050,
        "db": "demo"
    }

    connector = hadoop.impala_handler()
    connector.connect(**db_info)
    data = connector.select("select * from bearing order by idx_date limit 50")
    print(data)

    connector.close()
