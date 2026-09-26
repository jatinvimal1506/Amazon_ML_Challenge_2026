import logging
import re
import sys
import unicodedata

import pandas as pd

from business_entity_resolution.utils.exception import (
    CustomException
)

class DataTransformation:
    def __init__(self):
        pass

    @staticmethod
    def normalize_text(value):
        if pd.isna(value):
            return value

        value = str(value)

        #unicode normalisation
        value = unicodedata.normalize("NFKC",value) 
        value = value.lower()

        # Replace newlines/tabs with spaces
        value = re.sub(
            r"\s+",
            " ",
            value
        )#remove tabs and newlines and replace with spaces

        value = re.sub(
            r"[^\w\s]",
            " ",
            value
        )#punctutaion remove karna 


        value = re.sub(
            r"\s+",
            " ",
            value
        )#remove extra spaces

        return value.strip()

    def add_missing_flags(self, df):
        df["name_missing"] = (df["business_name"].isna()).astype(int)
        df["address_missing"] = (df["business_address"].isna()).astype(int)

        return df

    def clean_business_name(self, df):
        df["business_name_clean"] = (df["business_name"].apply(self.normalize_text))

        return df

    def clean_address(self, df):
        df["business_address_clean"] = (df["business_address"].apply(self.normalize_text))

        return df

    def transform_source(self, df):

        logging.info(
            "Starting data transformation."
        )

        try:
            df = df.copy()
            df = self.add_missing_flags(df)
            df = self.clean_business_name(df)
            df = self.clean_address(df)

            logging.info(
                "Data transformation completed successfully."
            )

            return df

        except Exception as e:
            logging.error(
                "Error occurred during data transformation."
            )

            raise CustomException(
                e,
                sys
            )