# -*- coding: utf-8 -*-
from RDA.preprocessing import symbolization
from RDA.analysis import ruleDiscovery
from RDA.databases import hadoop
import numpy as np
import pandas as pd


if __name__ == "__main__":
    db_info = {
        "host": "192.168.1.103",
        "port": 21050,
        "db": "demo"
    }
    connector = hadoop.impala_handler()
    connector.connect(**db_info)
    all_data = connector.select('select * from kddcup limit 100;')
    features = all_data.columns.tolist()
    s1_data_norm = symbolization.z_score_norm(all_data[features[4:-1]].to_numpy()[0:10])
    s2_data_norm = symbolization.z_score_norm(all_data[features[4:-1]].to_numpy()[5:15])
    connector.close()

    n_segmentation = int(s1_data_norm.shape[0] / 2)
    n_alphabet = 3

    paa_s1 = symbolization.paa(s1_data_norm, n_seg=n_segmentation)
    sax_s1 = symbolization.sax(paa_s1, n_alphabet)
    paa_s2 = symbolization.paa(s2_data_norm, n_seg=n_segmentation)
    sax_s2 = symbolization.sax(paa_s2, n_alphabet)
    s_id, basket = symbolization.symbol_baskets(sax_s1, n_alphabet)

    s1_basket = symbolization.symbol_sequence_baskets(sax_s1, 1, n_alphabet)
    s2_basket = symbolization.symbol_sequence_baskets(sax_s2, 2, n_alphabet)

    kwargs = {"max_length": 2, "min_support": 0.1, "min_confidence": 0.5, "min_lift": 2}
    res = ruleDiscovery.association_rule_discovery(basket, **kwargs)
    cols = ['items', 'support', 'base', 'add', 'confidence', 'lift']
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
