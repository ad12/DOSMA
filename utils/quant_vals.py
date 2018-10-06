from enum import Enum


class QuantitativeValues(Enum):
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


def get_qv(id):
    for qv in QuantitativeValues:
        if qv.name.lower() == id or qv.value == id:
            return qv


if __name__ == '__main__':
    print(type(QuantitativeValues.T1_RHO.name))
    print(QuantitativeValues.T1_RHO.value == 1)
