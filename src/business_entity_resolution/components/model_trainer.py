import logging
import os
import sys

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from business_entity_resolution.utils.exception import (
    CustomException
)

class ModelTrainer:

    def __init__(self, artifacts_dir="artifacts"):

        self.artifacts_dir = artifacts_dir

        os.makedirs(
            self.artifacts_dir,
            exist_ok=True
        )

        self.model_path = os.path.join(
            self.artifacts_dir,
            "match_model.joblib"
        )

    @staticmethod
    def prepare_ground_truth(ground_truth):

        logging.info(
            "Preparing ground truth pairs."
        )

        try:

            ground_truth_pairs = (
                ground_truth[
                    [
                        "source1_entity_id",
                        "matched_entity_ids"
                    ]
                ]
                .dropna(
                    subset=["matched_entity_ids"]
                )
                .assign(
                    matched_entity_id=lambda df:
                    df["matched_entity_ids"].str.split(",")
                )
                .explode(
                    "matched_entity_id"
                )
            )

            ground_truth_pairs[
                "matched_entity_id"
            ] = (
                ground_truth_pairs[
                    "matched_entity_id"
                ]
                .str.strip()
            )

            ground_truth_pairs = (
                ground_truth_pairs[
                    ground_truth_pairs[
                        "matched_entity_id"
                    ].ne("")
                ]
            )

            return ground_truth_pairs[
                [
                    "source1_entity_id",
                    "matched_entity_id"
                ]
            ].drop_duplicates()

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    @staticmethod
    def create_labels(
        feature_data,
        ground_truth
    ):

        logging.info(
            "Creating training labels."
        )

        try:

            ground_truth_pairs = (
                ModelTrainer.prepare_ground_truth(
                    ground_truth
                )
            )

            data = feature_data.copy()

            data = data.merge(
                ground_truth_pairs.assign(
                    label=1
                ),
                on=[
                    "source1_entity_id",
                    "matched_entity_id"
                ],
                how="left"
            )

            data["label"] = (
                data["label"]
                .fillna(0)
                .astype(int)
            )

            return data

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    @staticmethod
    def select_feature_columns(data):
        feature_columns = [
            "name_similarity",
            "address_similarity",
            "country_same",
            "name_missing_s1",
            "name_missing_s2",
            "address_missing_s1",
            "address_missing_s2"
        ]

        available_columns = [
            column
            for column in feature_columns
            if column in data.columns
        ]

        return available_columns

    @staticmethod
    def control_negative_samples(
        data,
        negative_ratio=3,
        random_state=42
    ):

        positives = data[
            data["label"] == 1
        ]

        negatives = data[
            data["label"] == 0
        ]

        if len(positives) == 0:
            raise ValueError(
                "No positive training pairs found."
            )

        max_negatives = (
            len(positives)
            * negative_ratio
        )

        if len(negatives) > max_negatives:

            negatives = negatives.sample(
                n=max_negatives,
                random_state=random_state
            )

        balanced_data = pd.concat(
            [
                positives,
                negatives
            ],
            ignore_index=True
        )

        return balanced_data.sample(
            frac=1,
            random_state=random_state
        ).reset_index(
            drop=True
        )

    def train_model(
        self,
        feature_data,
        ground_truth
    ):

        logging.info(
            "Starting model training."
        )

        try:

            data = self.create_labels(
                feature_data,
                ground_truth
            )

            feature_columns = (
                self.select_feature_columns(
                    data
                )
            )

            if not feature_columns:
                raise ValueError(
                    "No feature columns found."
                )

            data = self.control_negative_samples(
                data
            )

            X = data[
                feature_columns
            ].fillna(0)

            y = data["label"]

            logging.info(
                f"Training samples: {len(X)}"
            )

            logging.info(
                f"Positive samples: {(y == 1).sum()}"
            )

            logging.info(
                f"Negative samples: {(y == 0).sum()}"
            )

            model = HistGradientBoostingClassifier(
                learning_rate=0.08,
                max_iter=150,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42
            )

            model.fit(
                X,
                y
            )

            joblib.dump(
                {
                    "model": model,
                    "feature_columns": feature_columns
                },
                self.model_path
            )

            logging.info(
                f"Model saved at: {self.model_path}"
            )

            return (
                model,
                feature_columns
            )

        except Exception as e:

            logging.error(
                "Error occurred during model training."
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":
    print("ModelTrainer component loaded successfully.")