# -*- coding: utf-8 -*-
from RDA.preprocessing import normalization as normalizer
from common.databases import hadoop
from RDA.statistics import mspc

if __name__ == "__main__":
    db_info = {
        "host": "192.168.1.103",
        "port": 21050,
        "db": "demo"
    }

    connector = hadoop.impala_handler()
    connector.connect(**db_info)
    true = connector.select('select * from bearing '
                            'where idx_date < "2004-02-16 03:12:39" order by idx_date asc ;')
    cols = true.columns.tolist()
    data = connector.select('select * from bearing where idx_date >= "2004-02-16 03:12:39";')
    all_data = connector.select('select * from bearing order by idx_date asc limit 10;')
    connector.close()

    scaler = normalizer.standard_scaler(data[cols[1:]])
    scaled = scaler.transform(data[cols[1:]])

    print(true.corr(method='pearson'))
    print(data.corr(method='spearman'))
    print(true.corr(method='kendall'))

    _, _, lcl, ucl = mspc.tsquare_single(true[cols[1:]][8:], 4)
    all_score, _, _, _ = mspc.tsquare_single(data[cols[1:]], 4)
    decompose_score = mspc.tsquare_decomposition(data, cols[1:])
    residual = mspc.tsquare_residual(all_score, decompose_score)
    err_idx = mspc.ano_score_idx(all_score, lcl, ucl)

    a, b, c, d = [], [], [], []
    for i in err_idx:
        a.append(residual[0][i])
        b.append(residual[1][i])
        c.append(residual[2][i])
        d.append(residual[3][i])

    print(a)
    print(b)
    print(c)
    print(d)

    import numpy as np

    print(np.sum(a), np.sum(b), np.sum(c), np.sum(d))

    # _lcl = [lcl for i in range(all_score.size)]
    # _ucl = [ucl for i in range(all_score.size)]

    # plt.plot(all_score, c='black')
    # plt.plot(_lcl, c='red')
    # plt.plot(_ucl, c='red')

    # all_error = np.array([all_score[i] for i in err_idx])
    # line1, = plt.plot(all_error, c='black', label='all')
    # line2, = plt.plot(a, c='red', label='bearing1')
    # line3, = plt.plot(b, c='orange', label='bearing2')
    # line4, = plt.plot(c, c='yellow', label='bearing3')
    # line5, = plt.plot(d, c='green', label='bearing4')
    #
    # plt.legend([line1, line2, line3, line4, line5], ['All', 'bearing1', 'bearing2', 'bearing3', 'bearing4'])
    #
    # plt.show()