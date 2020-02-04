# -*- coding: utf-8 -*-
from apyori import apriori


def association_rule_discovery(basket, **kwargs):
    support = kwargs.get('min_support', 0.1)
    confidence = kwargs.get('min_confidence', 0.0)
    lift = kwargs.get('min_lift', 0.0)
    length = kwargs.get('max_length', None)

    rules = list(apriori(basket, min_support=support, min_confidence=confidence, min_lift=lift, max_length=length))

    res = []
    for line in rules:
        for item in line[2]:
            res.append([list(item[0]), "->", list(item[1]), " confi :", item[2], " lift :", item[3]])

    return res


def sequential_rule_discovery():
    pass
