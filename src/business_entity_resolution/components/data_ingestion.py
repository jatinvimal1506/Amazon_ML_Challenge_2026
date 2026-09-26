import logging
import sys
from dataclasses import dataclass

import pandas as pd

from business_entity_resolution.config.configuration import (
    ConfigurationManager
)

from business_entity_resolution.utils.exception import (
    CustomException
)

@dataclass
class DataIngestionConfig:
    train_source1_path: str
    train_source2_path: str
    train_source3_path: str
    train_ground_truth_path: str

class DataIngestion:

    def __init__(self):
        self.config = ConfigurationManager()

        self.ingestion_config = DataIngestionConfig(
            train_source1_path=self.config.train_source1,
            train_source2_path=self.config.train_source2,
            train_source3_path=self.config.train_source3,
            train_ground_truth_path=self.config.train_ground_truth
        )

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion.")

        try:
            source1 = pd.read_csv(
                self.ingestion_config.train_source1_path,
                sep="\t"
            )

            source2 = pd.read_csv(
                self.ingestion_config.train_source2_path,
                sep="\t"
            )

            source3 = pd.read_csv(
                self.ingestion_config.train_source3_path,
                sep="\t"
            )

            ground_truth = pd.read_csv(
                self.ingestion_config.train_ground_truth_path,
                sep="\t"
            )

            logging.info("All training datasets loaded successfully.")

            logging.info(f"Source 1 shape: {source1.shape}")
            logging.info(f"Source 2 shape: {source2.shape}")
            logging.info(f"Source 3 shape: {source3.shape}")
            logging.info(f"Ground truth shape: {ground_truth.shape}")

            return (
                source1,
                source2,
                source3,
                ground_truth
            )

        except Exception as e:
            logging.error("Error occurred during data ingestion.")

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":
    ingestion = DataIngestion()

    (
        source1,
        source2,
        source3,
        ground_truth
    ) = ingestion.initiate_data_ingestion()

    print("Source 1:", source1.shape)
    print("Source 2:", source2.shape)
    print("Source 3:", source3.shape)
    print("Ground Truth:", ground_truth.shape)