from enum import Enum


class QuantitativeValue(Enum):
    T1_RHO = 1
    T2 = 2
    T2_STAR = 3


if __name__ == '__main__':
    print(type(QuantitativeValue.T1_RHO.name))

    for i in QuantitativeValue:
        print(i)