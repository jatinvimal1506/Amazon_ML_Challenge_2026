import logging
import sys

import pandas as pd

from business_entity_resolution.utils.exception import (
    CustomException
)

class DataValidation:
    def __init__(self,source1,source2,source3,ground_truth):
        self.source1 = source1
        self.source2 = source2
        self.source3 = source3
        self.ground_truth = ground_truth

    def validate_columns(self,dataframe,expected_columns,dataframe_name):
        missing_columns = [
            column
            for column in expected_columns
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{dataframe_name} is missing columns: "
                f"{missing_columns}"
            )

    def validate_not_empty(self,dataframe,dataframe_name):
        if dataframe.empty:
            raise ValueError(
                f"{dataframe_name} is empty."
            )

    def initiate_data_validation(self):
        logging.info(
            "Starting data validation."
        )
        try:
            expected_source_columns = [
                "entity_id",
                "business_name",
                "business_address",
                "country"
            ]

            expected_ground_truth_columns = [
                "source1_entity_id",
                "matched_entity_ids"
            ]

            self.validate_columns(
                self.source1,
                expected_source_columns,
                "Source 1"
            )

            self.validate_columns(
                self.source2,
                expected_source_columns,
                "Source 2"
            )

            self.validate_columns(
                self.source3,
                expected_source_columns,
                "Source 3"
            )

            self.validate_columns(
                self.ground_truth,
                expected_ground_truth_columns,
                "Ground Truth"
            )

            self.validate_not_empty(
                self.source1,
                "Source 1"
            )

            self.validate_not_empty(
                self.source2,
                "Source 2"
            )

            self.validate_not_empty(
                self.source3,
                "Source 3"
            )

            self.validate_not_empty(
                self.ground_truth,
                "Ground Truth"
            )

            logging.info(
                "Data validation completed successfully."
            )

            return True

        except Exception as e:
            logging.error(
                "Error occurred during data validation."
            )

            raise CustomException(
                e,
                sys
            )