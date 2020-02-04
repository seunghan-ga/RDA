# -*- coding: utf-8 -*-
from RDA.preprocessing import symbolization
from RDA.analysis import ruleDiscovery
from RDA.databases import hadoop
# from apyori import apriori
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf


features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "target"]


if __name__ == "__main__":
    kddcup_data = pd.read_csv("samples/kddcup.data.corrected")
    kddcup_data.columns = features

    # db_info = {
    #     "host": "192.168.1.103",
    #     "port": 21050,
    #     "db": "demo"
    # }
    #
    # connector = hadoop.impala_handler()
    # connector.connect(**db_info)
    # all_data = connector.select('select * from bearing order by idx_date asc;')
    # cols = all_data.columns.tolist()
    # connector.close()

    # data_norm = symbolization.z_score_norm(all_data[cols[1:]].to_numpy())
    data_norm = symbolization.z_score_norm(kddcup_data[features[4:-1]].to_numpy()[:10])

    n_segmentation = int(data_norm.shape[0] / 2)
    n_alphabet = 3

    paa = symbolization.paa(data_norm, n_seg=n_segmentation)
    sax = symbolization.sax(paa, n_alphabet)
    id, basket = symbolization.symbol_baskets(sax, n_alphabet)

    symbolization.symbol_sequence_baskets(sax, 1, n_alphabet)

    kwargs = {"max_length": 2, "min_support": 0.1, "min_confidenc": 0.5, "min_lift": 2}
    res = ruleDiscovery.association_rule_discovery(basket, **kwargs)
    # for i in res:
    #     print(i)

