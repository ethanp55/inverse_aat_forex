from dataclasses import dataclass
from typing import List


@dataclass
class CreditFeatures:
    limit_bal: float  # Amount of given credit
    gender: int  # Gender
    education: int  # Education level
    marriage: int  # Marital status
    age: int  # Age (in years)
    pay_0: int  # Payment delay (in months) in September 2005
    pay_2: int  # Payment delay (in months) in August 2005
    pay_3: int  # Payment delay (in months) in July 2005
    pay_4: int  # Payment delay (in months) in June 2005
    pay_5: int  # Payment delay (in months) in May 2005
    pay_6: int  # Payment delay (in months) in April 2005
    bill_amt1: float  # Amount on bill in September 2005
    bill_amt2: float  # Amount on bill in August 2005
    bill_amt3: float  # Amount on bill in July 2005
    bill_amt4: float  # Amount on bill in June 2005
    bill_amt5: float  # Amount on bill in May 2005
    bill_amt6: float  # Amount on bill in April 2005
    pay_amt1: float  # Amount of payment in September 2005
    pay_amt2: float  # Amount of payment in August 2005
    pay_amt3: float  # Amount of payment in July 2005
    pay_amt4: float  # Amount of payment in June 2005
    pay_amt5: float  # Amount of payment in May 2005
    pay_amt6: float  # Amount of payment in April 2005

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]

    def get_names(self) -> List[str]:
        attribute_names = self.__annotations__.keys()

        return list(attribute_names)


class CreditAssumptions:
    def __init__(self, credit_features: CreditFeatures, pred: int) -> None:
        # Assumptions
        self.limit_bal_below_avg = credit_features.limit_bal < 167484.323
        self.limit_bal_above_avg = not self.limit_bal_below_avg
        self.gender = credit_features.gender
        self.university_edu_and_higher = credit_features.education <= 2
        self.low_edu = not self.university_edu_and_higher
        self.married = credit_features.marriage == 1
        self.single = credit_features.marriage == 2
        self.age_below_avg = credit_features.age < 35.4855
        self.age_above_avg = not self.age_below_avg
        pay_avg = sum([credit_features.pay_0, credit_features.pay_2, credit_features.pay_3, credit_features.pay_4,
                       credit_features.pay_5, credit_features.pay_6]) / 6
        self.on_time = pay_avg <= 0
        self.slight_delay = 0 < pay_avg < 4
        self.major_delay = pay_avg > 4
        self.pay_amt1_full = credit_features.pay_amt1 > credit_features.bill_amt1
        self.pay_amt2_full = credit_features.pay_amt2 > credit_features.bill_amt2
        self.pay_amt3_full = credit_features.pay_amt3 > credit_features.bill_amt3
        self.pay_amt4_full = credit_features.pay_amt4 > credit_features.bill_amt4
        self.pay_amt5_full = credit_features.pay_amt5 > credit_features.bill_amt5
        self.pay_amt6_full = credit_features.pay_amt6 > credit_features.bill_amt6
        bill_amt_avg = sum([credit_features.bill_amt1, credit_features.bill_amt2, credit_features.bill_amt3,
                            credit_features.bill_amt4, credit_features.bill_amt5, credit_features.bill_amt6]) / 6
        pay_amt_avg = sum([credit_features.pay_amt1, credit_features.pay_amt2, credit_features.pay_amt3,
                           credit_features.pay_amt4, credit_features.pay_amt5, credit_features.pay_amt6]) / 6
        self.paid_in_full = pay_amt_avg >= bill_amt_avg

        # Label
        self.prediction = pred

    def create_aat_tuple(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [self.__getattribute__(name) for name in attribute_names]

        return tup

    def assumption_names(self) -> List[str]:
        attribute_names = self.__dict__.keys()
        filtered_names = [name for name in attribute_names if name != 'prediction']

        assert len(filtered_names) == len(attribute_names) - 1

        return filtered_names
