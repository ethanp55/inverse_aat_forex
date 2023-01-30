from dataclasses import dataclass
from typing import List


@dataclass
class InsuranceFeatures:
    months_as_customer: int  # Number of months as a customer with the insurance company
    age: int  # Customer's age
    policy_csl: float  # CSL
    policy_deductible: int  # Deductible
    policy_annual_premium: float  # Premium
    umbrella_limit: int  # Limit
    capital_gains: int  # Customer's capital gains
    capital_loss: int  # Customer's capital loss
    incident_hour_of_the_day: int  # Hour when the incident occurred
    number_of_vehicles_involved: int  # How many vehicles were involved in the incident
    bodily_injuries: int  # Number of bodily injuries in the incident
    witnesses: int  # Number of incident witnesses
    total_claim_amount: int  # How much the insurance company will pay in total
    injury_claim: int  # How much the insurance company will pay for injuries
    property_claim: int  # How much the insurance company will pay for property damages
    vehicle_claim: int  # How much the insurance company will pay for vehicular damages
    auto_year: int  # Year the car in question was made
    insured_sex_female: int  # Bool indicating female gender
    insured_sex_male: int  # Bool indicating male gender
    insured_education_level_associate: int  # Bool indicating associate degree
    insured_education_level_college: int  # Bool indicating college degree
    insured_education_level_high_school: int  # Bool indicating high school degree
    insured_education_level_jd: int  # Bool indicating JD degree
    insured_education_level_md: int  # Bool indicating MD degree
    insured_education_level_masters: int  # Bool indicating Masters degree
    insured_education_level_phd: int  # Bool indicating PhD degree
    insured_occupation_adm_clerical: int  # Bool indicating job as clerk
    insured_occupation_armed_forces: int  # Bool indicating job in armed forces
    insured_occupation_craft_repair: int  # Bool indicating job in crafts/repair
    insured_occupation_exec_manage: int  # Bool indicating job as executive/manager
    insured_occupation_farm_fish: int  # Bool indicating job as farmer/fisher
    insured_occupation_handlers_cleaners: int  # Bool indicating job in handling/cleaning
    insured_occupation_machine_op: int  # Bool indicating job in machine operations
    insured_occupation_other_service: int  # Bool indicating job in other services
    insured_occupation_house_servant: int  # Bool indicating job as private house servant
    insured_occupation_specialty: int  # Bool indicating job in specialty field
    insured_occupation_protective: int  # Bool indicating job in protective services
    insured_occupation_sales: int  # Bool indicating job in sales
    insured_occupation_tech: int  # Bool indicating job in tech support
    insured_occupation_transport: int  # Bool indicating job in transportation/moving
    insured_relationship_husband: int  # Bool indicating relationship status of "husband"
    insured_relationship_no_family: int  # Bool indicating relationship status of having no family
    insured_relationship_other_relative: int  # Bool indicating relationship status of having distance relatives
    insured_relationship_own_child: int  # Bool indicating relationship status of having a child
    insured_relationship_unmarried: int  # Bool indicating relationship status of being unmarried
    insured_relationship_wife: int  # Bool indicating relationship status of "wife"
    incident_type_multi_vehicle: int  # Bool indicating incident with multiple vehicles
    incident_type_parked_car: int  # Bool indicating incident with a parked car
    incident_type_single_vehicle: int  # Bool indicating incident with a single vehicle
    incident_type_theft: int  # Bool indicating incident with vehicular theft
    incident_type_major_damage: int  # Bool indicating incident with major damage
    incident_type_minor_damage: int  # Bool indicating incident with minor damage
    incident_type_total_loss: int  # Bool indicating incident with a total loss
    incident_type_trivial_damage: int  # Bool indicating incident with trivial damage
    police_report_available_unknown: int  # Bool indicating confusion about available police report
    police_report_available_no: int  # Bool indicating no available police report
    police_report_available_yes: int  # Bool indicating an available police report

    def get_values(self) -> List[float]:
        attribute_names = self.__annotations__.keys()
        return [self.__getattribute__(field_name) for field_name in attribute_names]

    def get_names(self) -> List[str]:
        attribute_names = self.__annotations__.keys()

        return list(attribute_names)


class InsuranceAssumptions:
    def __init__(self, insurance_features: InsuranceFeatures, pred: int) -> None:
        # Assumptions
        self.new_customer = insurance_features.months_as_customer < 203.954
        self.young_customer = insurance_features.age < 38.948
        self.small_csl = insurance_features.policy_csl < 0.5
        self.small_deductible = insurance_features.policy_deductible < 1136
        self.small_premium = insurance_features.policy_annual_premium < 1256.40615
        self.negative_limit = insurance_features.umbrella_limit < 0
        self.small_limit = insurance_features.umbrella_limit < 1101000
        self.small_capital_gains = insurance_features.capital_gains < 25126.1
        self.large_capital_losses = insurance_features.capital_loss < -111100
        self.early_hours = insurance_features.incident_hour_of_the_day < 11.644
        self.few_vehicles_involved = insurance_features.number_of_vehicles_involved < 1.839
        self.bodily_injuries = insurance_features.bodily_injuries > 0
        self.witnesses = insurance_features.witnesses > 0
        self.multiple_witnesses = insurance_features.witnesses > 1
        self.small_total_claim = insurance_features.total_claim_amount < 52761.94
        self.small_injury_claim = insurance_features.injury_claim < 7433.42
        self.small_property_claim = insurance_features.property_claim < 7399.57
        self.small_vehicle_claim = insurance_features.vehicle_claim < 37928.95
        self.old_car = insurance_features.auto_year < 2000
        self.new_car = insurance_features.auto_year > 2010
        self.male = insurance_features.insured_sex_male
        self.low_edu = insurance_features.insured_education_level_high_school or insurance_features.insured_education_level_associate
        self.medium_edu = insurance_features.insured_education_level_college
        self.high_edu = insurance_features.insured_education_level_md or \
            insurance_features.insured_education_level_jd or insurance_features.insured_education_level_phd or \
            insurance_features.insured_education_level_masters
        self.specialty_edu = insurance_features.insured_education_level_md or \
            insurance_features.insured_education_level_jd
        self.basic_job = insurance_features.insured_occupation_armed_forces or \
            insurance_features.insured_occupation_protective or insurance_features.insured_occupation_transport or \
            insurance_features.insured_occupation_machine_op or \
            insurance_features.insured_occupation_handlers_cleaners or \
            insurance_features.insured_occupation_house_servant or insurance_features.insured_occupation_craft_repair \
            or insurance_features.insured_occupation_farm_fish
        self.medium_job = insurance_features.insured_occupation_adm_clerical or \
            insurance_features.insured_occupation_sales or insurance_features.insured_occupation_specialty or \
            insurance_features.insured_occupation_other_service or insurance_features.insured_occupation_tech
        self.adv_job = insurance_features.insured_occupation_exec_manage
        self.married = insurance_features.insured_relationship_husband or \
            insurance_features.insured_relationship_wife or not insurance_features.insured_relationship_unmarried
        self.has_family = insurance_features.insured_relationship_own_child or \
            insurance_features.insured_relationship_other_relative or not \
            insurance_features.insured_relationship_no_family
        self.collision = insurance_features.incident_type_multi_vehicle or \
            insurance_features.incident_type_single_vehicle
        self.major_damage = insurance_features.incident_type_major_damage or insurance_features.incident_type_total_loss
        self.police_report = insurance_features.police_report_available_yes

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
