# -*- coding: utf-8 -*-
from RDA.preprocessing import symbolization
from RDA.analysis import ruleDiscovery
from RDA.databases import hadoop
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
    # db_info = {
    #     "host": "192.168.1.103",
    #     "port": 21050,
    #     "db": "demo"
    # }
    # connector = hadoop.impala_handler()
    # connector.connect(**db_info)
    # all_data = connector.select('select * from bearing order by idx_date asc;')
    # cols = all_data.columns.tolist()
    # data_norm = symbolization.z_score_norm(all_data[cols[1:]].to_numpy())
    # connector.close()

    kddcup_data = pd.read_csv("samples/kddcup.data.corrected")
    kddcup_data.columns = features
    s1_data_norm = symbolization.z_score_norm(kddcup_data[features[4:-1]].to_numpy()[:10])
    s2_data_norm = symbolization.z_score_norm(kddcup_data[features[4:-1]].to_numpy()[5:15])

    n_segmentation = int(s1_data_norm.shape[0] / 2)
    n_alphabet = 3

    paa_s1 = symbolization.paa(s1_data_norm, n_seg=n_segmentation)
    sax_s1 = symbolization.sax(paa_s1, n_alphabet)
    paa_s2 = symbolization.paa(s2_data_norm, n_seg=n_segmentation)
    sax_s2 = symbolization.sax(paa_s2, n_alphabet)
    s_id, basket = symbolization.symbol_baskets(sax_s1, n_alphabet)

    s1_basket = symbolization.symbol_sequence_baskets(sax_s1, 1, n_alphabet)
    s2_basket = symbolization.symbol_sequence_baskets(sax_s2, 2, n_alphabet)

    kwargs = {"max_length": 2, "min_support": 0.1, "min_confidenc": 0.5, "min_lift": 2}
    res = ruleDiscovery.association_rule_discovery(basket, **kwargs)
    cols = ['base', 'add', 'confidence', 'lift']
    df1 = pd.DataFrame(res, columns=cols)

    s_baskets = np.concatenate((s1_basket, s2_basket)).tolist()
    s_list = symbolization.flatten_unique(s_baskets)
    s_baskets_convert = symbolization.idx_value_convert(s_baskets, s_list)

    kwargs = {"min_support": 0.3, "parse": True, "max_length": 2}
    res = ruleDiscovery.sequential_rule_discovery(s_baskets_convert, s_list, **kwargs)
    cols = ['Occurs', 'Accum', 'Support', 'Confid', 'Lift', 'Sequence']
    df2 = pd.DataFrame(res, columns=cols)

    print(df1)
    print(df2)
