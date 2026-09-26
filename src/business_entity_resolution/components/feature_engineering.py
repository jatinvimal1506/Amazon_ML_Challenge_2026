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

    #similarity
    @staticmethod
    def calculate_similarity(
        value1,
        value2
    ):

        if pd.isna(value1) or pd.isna(value2):
            return 0.0

        value1 = str(value1)
        value2 = str(value2)

        if not value1 or not value2:
            return 0.0

        return float(
            fuzz.ratio(
                value1,
                value2
            )
        )


    # COUNTRY MATCH
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


    #create features
    def create_features(
        self,
        candidates
    ):

        logging.info(
            "Starting feature engineering."
        )

        try:

            df = candidates.copy()
            # Detect source column automatically
            source_column = None

            if "business_name_clean_s2" in df.columns:
                source_column = "s2"

            elif "business_name_clean_s3" in df.columns:
                source_column = "s3"

            else:
                raise ValueError(
                    "Could not identify S2 or S3 candidate columns."
                )

            # Column namw
            s1_name_column = (
                "business_name_clean_s1"
            )

            source_name_column = (
                f"business_name_clean_{source_column}"
            )

            s1_address_column = (
                "business_address_clean_s1"
            )

            source_address_column = (
                f"business_address_clean_{source_column}"
            )

            s1_country_column = (
                "country_s1"
            )

            source_country_column = (
                f"country_{source_column}"
            )

            s1_name_raw_column = (
                "business_name_s1"
            )

            source_name_raw_column = (
                f"business_name_{source_column}"
            )

            s1_address_raw_column = (
                "business_address_s1"
            )

            source_address_raw_column = (
                f"business_address_{source_column}"
            )

            # -------------------------------------------------
            # Name similarity
            # -------------------------------------------------

            df["name_similarity"] = [
                self.calculate_similarity(
                    name1,
                    name2
                )
                for name1, name2 in zip(
                    df[s1_name_column],
                    df[source_name_column]
                )
            ]

            # -------------------------------------------------
            # Address similarity
            # -------------------------------------------------

            df["address_similarity"] = [
                self.calculate_similarity(
                    address1,
                    address2
                )
                for address1, address2 in zip(
                    df[s1_address_column],
                    df[source_address_column]
                )
            ]

            # -------------------------------------------------
            # Country match
            # -------------------------------------------------

            df["country_same"] = [
                self.calculate_country_match(
                    country1,
                    country2
                )
                for country1, country2 in zip(
                    df[s1_country_column],
                    df[source_country_column]
                )
            ]

            # -------------------------------------------------
            # Name missing flags
            # -------------------------------------------------

            df["name_missing_s1"] = (
                df[s1_name_raw_column]
                .isna()
                .astype(int)
            )

            df["name_missing_source"] = (
                df[source_name_raw_column]
                .isna()
                .astype(int)
            )

            # -------------------------------------------------
            # Address missing flags
            # -------------------------------------------------

            df["address_missing_s1"] = (
                df[s1_address_raw_column]
                .isna()
                .astype(int)
            )

            df["address_missing_source"] = (
                df[source_address_raw_column]
                .isna()
                .astype(int)
            )

            # -------------------------------------------------
            # Store source information
            # -------------------------------------------------

            df["candidate_source"] = source_column

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

    print(
        "FeatureEngineering component loaded successfully."
    )