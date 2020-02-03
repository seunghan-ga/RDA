# -*- coding: utf-8 -*-
from RDA.preprocessing import symbolization
from RDA.databases import hadoop
from apyori import apriori
import numpy as np
import pandas as pd


features = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
            "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
            "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
            "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate", "target"]


if __name__ == "__main__":
    # kddcup_data = pd.read_csv("samples/kddcup.data.corrected")
    # kddcup_data.columns = features
    #
    # data_norm = symbolization.z_score_norm(kddcup_data[features[4:-1]].to_numpy()[:50])
    db_info = {
        "host": "192.168.1.103",
        "port": 21050,
        "db": "demo"
    }

    connector = hadoop.impala_handler()
    connector.connect(**db_info)
    all_data = connector.select('select * from bearing order by idx_date asc;')
    cols = all_data.columns.tolist()
    connector.close()

    data_norm = symbolization.z_score_norm(all_data[cols[1:]].to_numpy())
    n_segmentation = int(data_norm.shape[0] / 2)
    n_alphabet = 10

    paa = symbolization.paa(data_norm, n_seg=n_segmentation)
    sax = symbolization.sax(paa, n_alphabet)
    id, basket = symbolization.symbol_baskets(sax, n_alphabet)

    # print(np.array(basket))
    # res = list(apriori(basket))
    # for item in res:
    #     print(item)

    print("========================================================================================")

    res = list(apriori(basket, max_length=3, min_lift=2))
    print(len(res))

    for idx in range(len(res)):
        line = res[idx]
        if len(line[0]) > 1:
            print("basket id : ", id[idx])
            print("items :", list(line[0]), " support :", line[1])

            for item in line[2][1:]:
                print(list(item[0]), " -> ", list(item[1]), " confidence :", item[2], " lift :", item[3])
            print("========================================================================================")
