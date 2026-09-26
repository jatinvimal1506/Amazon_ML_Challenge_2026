import logging
import sys

import pandas as pd
from rapidfuzz import fuzz

from business_entity_resolution.utils.exception import (
    CustomException
)

class FeatureEngineering:
    def __init__(self):
        pass

    @staticmethod
    def calculate_similarity(value1, value2):
        if pd.isna(value1) or pd.isna(value2):
            return 0.0

        value1 = str(value1)
        value2 = str(value2)

        if not value1 or not value2:
            return 0.0

        return fuzz.ratio(
            value1,
            value2
        )

    @staticmethod
    def calculate_country_match(
        country1,
        country2
    ):

        if pd.isna(country1) or pd.isna(country2):
            return 0

        return int(
            str(country1).strip().lower()
            == str(country2).strip().lower()
        )

    def create_features(
        self,
        candidates
    ):

        logging.info(
            "Starting feature engineering."
        )

        try:

            df = candidates.copy()

            # Name similarity
            df["name_similarity"] = [
                self.calculate_similarity(
                    name1,
                    name2
                )
                for name1, name2 in zip(
                    df["business_name_clean_s1"],
                    df["business_name_clean_s2"]
                )
            ]
            # Address similarity

            df["address_similarity"] = [
                self.calculate_similarity(
                    address1,
                    address2
                )
                for address1, address2 in zip(
                    df["business_address_clean_s1"],
                    df["business_address_clean_s2"]
                )
            ]

            # Country match

            df["country_same"] = [
                self.calculate_country_match(
                    country1,
                    country2
                )
                for country1, country2 in zip(
                    df["country_s1"],
                    df["country_s2"]
                )
            ]
            # Missing-value indicators


            df["name_missing_s1"] = (
                df["business_name_s1"]
                .isna()
                .astype(int)
            )

            df["name_missing_s2"] = (
                df["business_name_s2"]
                .isna()
                .astype(int)
            )

            df["address_missing_s1"] = (
                df["business_address_s1"]
                .isna()
                .astype(int)
            )

            df["address_missing_s2"] = (
                df["business_address_s2"]
                .isna()
                .astype(int)
            )

            logging.info(
                "Feature engineering completed successfully."
            )

            return df

        except Exception as e:
            logging.error(
                "Error occurred during feature engineering."
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":
    print("FeatureEngineering component loaded successfully.")