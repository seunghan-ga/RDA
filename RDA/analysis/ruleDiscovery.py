# -*- coding: utf-8 -*-
from apyori import apriori
from pycspade.helpers import spade


def association_rule_discovery(basket, **kwargs):
    """
    association rule discovery
    :param basket: symbol baskets
    :param kwargs: apriori arguments
    :return: rule discovery result
    """
    support = kwargs.get('min_support', 0.1)
    confidence = kwargs.get('min_confidence', 0.0)
    lift = kwargs.get('min_lift', 0.0)
    length = kwargs.get('max_length', None)

    rules = list(apriori(basket, min_support=support, min_confidence=confidence, min_lift=lift, max_length=length))

    res = []
    for line in rules:
        for item in line[2]:
            # base, add, confidence, lift
            res.append([list(item[0]), list(item[1]), item[2], item[3]])

    return res


def sequential_rule_discovery(s_basket, s_list, **kwargs):
    """
    sequential rule discovery
    :param s_basket: symbol sequence basket
    :param s_list: converted array (1N, symbol -> index)
    :param kwargs: spade arguments
    :return: rule discovery result
    """
    support = kwargs["min_support"]
    parse = kwargs["parse"]
    length = kwargs["max_length"]
    result = spade(data=s_basket, support=support, parse=parse, maxlen=length)

    res = []
    for mined_object in result['mined_objects']:
        # Occurs, Accum, Support, Confid, Lift, Sequence
        res.append([mined_object.noccurs,
                    mined_object.accum_occurs,
                    mined_object.noccurs / result['nsequences'],
                    mined_object.confidence if mined_object.confidence else 'N/A',
                    mined_object.lift if mined_object.lift else 'N/A',
                    [s_list[int(str(i).replace('(', '').replace(')', ''))] for i in mined_object.items]])

    return res
