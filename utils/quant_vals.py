from enum import Enum


class QuantitativeValues(Enum):
    """Enum of quantitative values that can be analyzed"""
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


def get_qv(id):
    """Find QuantitativeValue enum using id
    :param id: either enum number or name in lower case
    :return: Quantitative value corresponding to id

    :raise ValueError: if no QuantitativeValue corresponding to id found
    """
    for qv in QuantitativeValues:
        if qv.name.lower() == id or qv.value == id:
            return qv

    raise ValueError('Quantitative Value with name or id %s not found' % str(id))


if __name__ == '__main__':
    print(type(QuantitativeValues.T1_RHO.name))
    print(QuantitativeValues.T1_RHO.value == 1)
