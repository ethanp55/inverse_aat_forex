from dataclasses import dataclass
import numpy as np
import pickle
from sklearn.neighbors import KernelDensity
from typing import Optional, List


class CompasFeatureDistributions:
    def __init__(self, arry: np.array, file_path: str, load: Optional[bool] = None) -> None:
        if load is not None and load:
            self.distributions = self.load(file_path)

        else:
            self.distributions = [KernelDensity(kernel='gaussian', bandwidth=0.2).fit(arry[:, j].reshape(-1, 1))
                                  for j in range(arry.shape[-1])]
            self.save(file_path)

    def save(self, file_path: str) -> None:
        with open(f'{file_path}.pickle', 'wb') as f:
            pickle.dump(self.distributions, f)

    def load(self, file_path: str) -> List[KernelDensity]:
        return pickle.load(open(f'{file_path}.pickle', 'rb'))

    def calculate_probs(self, x: List[float]) -> List[float]:
        assert len(self.distributions) == len(x)

        probs = [np.exp(self.distributions[i].score_samples(np.array(x[i]).reshape(1, -1))[0]) for i in range(len(x))]

        return probs


@dataclass
class CompasDistributionAssumptions:
    # Assumptions/features
    age_prob: float
    priors_count_prob: float
    days_b_screening_arrest_prob: float
    is_recid_prob: float
    two_year_recid_prob: float
    c_charge_degree_f_prob: float
    c_charge_degree_m_prob: float
    race_african_american_prob: float
    race_asian_prob: float
    race_caucasian_prob: float
    race_hispanic_prob: float
    race_native_american_prob: float
    race_other_prob: float
    age_cat_25_45_prob: float
    age_cat_gt_45_prob: float
    age_cat_lt_25_prob: float
    gender_female_prob: float
    gender_male_prob: float

    # Label
    prediction: int

    def create_tuple(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [self.__getattribute__(name) for name in attribute_names]

        return tup

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]

    def assumption_names(self) -> List[str]:
        attribute_names = self.__dict__.keys()
        filtered_names = [name for name in attribute_names if name != 'prediction']

        assert len(filtered_names) == len(attribute_names) - 1

        return filtered_names
