from dataclasses import dataclass
from typing import List


@dataclass
class BaseballFeatures:
    g: int  # Number of games played
    pa: int  # Number of plate appearances
    hr: int  # Number of home runs
    r: int  # Number of runs scored
    rbi: int  # Number of runs batted in
    sb: int  # Number of stolen bases
    bb_percent: float  # Walk rate
    k_percent: float  # Strikeout rate
    iso: float  # Isolated power (only takes extra-base hits into account; basically is calculated as number of extra-base hits / number of bats, but with extra weight on triples and HRs)
    babip: float  # Rate of how often a ball in play counts as a hit
    avg: float  # Overall batting average
    obp: float  # On base percentage
    slg: float  # Batting average, but weighted by the type of hit (single = 1, double = 2, etc.)
    woba: float  # On base percentage, but weighted by how the player reached base
    bsr: float  # Number of runs scored a player/team should have had after taking into account steals, base running, etc.
    off: float  # Offensive statistic that combines a position player’s total context-neutral value at the plate and on the bases
    Def: float  # Player’s total defensive value relative to league average
    war: float  # How many more wins a player's worth than a replacement-level player at the same position

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]

    def get_names(self) -> List[str]:
        attribute_names = self.__annotations__.keys()

        return list(attribute_names)


class BaseballAssumptions:
    def __init__(self, baseball_features: BaseballFeatures, pred: float) -> None:
        # Assumptions
        # self.baseball_features = baseball_features
        n_games_in_season = 162
        self.less_than_third_of_season = baseball_features.g < (n_games_in_season / 3)
        self.more_than_half_of_season = baseball_features.g > (n_games_in_season * 0.5)
        self.more_than_three_fourths_of_season = baseball_features.g > (n_games_in_season * 0.75)
        avg_pa = baseball_features.pa / baseball_features.g
        self.above_avg_pa = avg_pa > 4
        self.above_avg_hr = baseball_features.hr > 20
        self.significant_hr = baseball_features.hr > 30
        self.extraordinary_hr = baseball_features.hr > 40
        self.legendary_hr = baseball_features.hr > 50
        self.above_avg_r = baseball_features.r > 20
        self.significant_r = baseball_features.r > 30
        self.extraordinary_r = baseball_features.r > 40
        self.above_avg_rbi = baseball_features.rbi > 20
        self.significant_rbi = baseball_features.rbi > 30
        self.extraordinary_rbi = baseball_features.rbi > 40
        self.above_avg_sb = baseball_features.sb > 2
        self.significant_sb = baseball_features.sb > 10
        self.above_avg_bb = baseball_features.bb_percent > 0.06
        self.significant_bb = baseball_features.bb_percent > 0.1
        self.extraordinary_bb = baseball_features.bb_percent > 0.15
        self.below_avg_k = baseball_features.k_percent < 0.25
        self.significant_k = baseball_features.k_percent < 0.15
        self.extraordinary_k = baseball_features.k_percent < 0.1
        self.above_avg_iso = baseball_features.iso > 0.08
        self.significant_iso = baseball_features.iso > 0.15
        self.above_avg_babip = baseball_features.babip > 0.18
        self.significant_babip = baseball_features.babip > 0.25
        self.above_avg_avg = baseball_features.avg > 0.18
        self.significant_avg = baseball_features.avg > 0.25
        self.extraordinary_avg = baseball_features.avg > 0.3
        self.legendary_avg = baseball_features.avg > 0.35
        self.above_avg_obp = baseball_features.obp > 0.2
        self.significant_obp = baseball_features.obp > 0.25
        self.extraordinary_obp = baseball_features.obp > 0.3
        self.legendary_obp = baseball_features.obp > 0.35
        self.above_avg_slg = baseball_features.slg > 0.3
        self.significant_slg = baseball_features.slg > 0.45
        self.extraordinary_slg = baseball_features.slg > 0.55
        self.above_avg_woba = baseball_features.woba > 0.3
        self.significant_woba = baseball_features.woba > 0.4
        self.above_avg_war = baseball_features.war > 1
        self.significant_war = baseball_features.war > 3
        self.extraordinary_war = baseball_features.war > 5

        # Label
        self.prediction = pred

    def create_aat_tuple(self) -> List[float]:
        attribute_names = self.__dict__.keys()
        tup = [self.__getattribute__(name) for name in attribute_names]

        # tup = self.baseball_features.get_values() + [self.prediction]

        return tup

    def assumption_names(self) -> List[str]:
        attribute_names = self.__dict__.keys()
        filtered_names = [name for name in attribute_names if name != 'prediction']

        assert len(filtered_names) == len(attribute_names) - 1

        return filtered_names

        # return self.baseball_features.get_names()
