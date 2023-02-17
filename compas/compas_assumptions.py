from dataclasses import dataclass
from typing import List


@dataclass
class CompasFeatures:
    age: int
    priors_count: int
    days_b_screening_arrest: float
    is_recid: int
    two_year_recid: int
    c_charge_degree_f: int
    c_charge_degree_m: int
    race_african_american: int
    race_asian: int
    race_caucasian: int
    race_hispanic: int
    race_native_american: int
    race_other: int
    age_cat_25_45: int
    age_cat_gt_45: int
    age_cat_lt_25: int
    gender_female: int
    gender_male: int

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]

    def get_names(self) -> List[str]:
        attribute_names = self.__annotations__.keys()

        return list(attribute_names)


class CompasAssumptions:
    def __init__(self, compas_features: CompasFeatures, pred: int) -> None:
        # Assumptions
        self.young = compas_features.age_cat_lt_25
        self.middle_aged = compas_features.age_cat_25_45
        self.old = compas_features.age_cat_gt_45
        self.few_priors = compas_features.priors_count < 3
        self.multiple_priors = 3 <= compas_features.priors_count <= 15
        self.several_priors = compas_features.priors_count > 15
        self.negative_days_b_screening_arrest = compas_features.days_b_screening_arrest < 0
        self.positive_days_b_screening_arrest = compas_features.days_b_screening_arrest > 0
        self.is_recid = compas_features.is_recid
        self.two_year_recid = compas_features.two_year_recid
        self.c_charge_degree_f = compas_features.c_charge_degree_f
        self.race_african_american = compas_features.race_african_american
        self.race_asian = compas_features.race_asian
        self.race_caucasian = compas_features.race_caucasian
        self.race_hispanic = compas_features.race_hispanic
        self.race_native_american = compas_features.race_native_american
        self.race_other = compas_features.race_other
        self.male = compas_features.gender_male

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
