import logging
import sys

import numpy as np
import pandas as pd

from business_entity_resolution.utils.exception import (
    CustomException
)

class ModelEvaluation:
    def __init__(self):
        pass

    @staticmethod
    def calculate_fbeta(
        precision,
        recall,
        beta=0.5
    ):

        if precision == 0 and recall == 0:
            return 0.0

        beta_squared = beta ** 2

        return (
            (1 + beta_squared)
            * precision
            * recall
        ) / (
            (beta_squared * precision)
            + recall
        )

    @staticmethod
    def calculate_macro_fbeta(
        data,
        threshold,
        beta=0.5
    ):
        """
        Calculate macro F0.5 at the S1 entity level.

        Each S1 entity is evaluated independently based on:
        - true matched entity IDs
        - predicted matched entity IDs

        If an S1 has no true matches:
            empty prediction -> score 1.0
            any prediction   -> score 0.0
        """

        scores = []

        for source1_id, group in data.groupby(
            "source1_entity_id"
        ):

            true_matches = set(
                group.loc[
                    group["label"] == 1,
                    "matched_entity_id"
                ]
            )

            predicted_matches = set(
                group.loc[
                    group["prediction_score"] >= threshold,
                    "matched_entity_id"
                ]
            )

            # Special case: true entity has no matches
            if len(true_matches) == 0:
                if len(predicted_matches) == 0:
                    scores.append(1.0)

                else:
                    scores.append(0.0)

                continue

            # Calculate precision / recall
            true_positives = len(
                true_matches.intersection(
                    predicted_matches
                )
            )

            false_positives = len(
                predicted_matches
                - true_matches
            )

            false_negatives = len(
                true_matches
                - predicted_matches
            )

            precision = (
                true_positives
                / (
                    true_positives
                    + false_positives
                )
                if (
                    true_positives
                    + false_positives
                ) > 0
                else 0.0
            )

            recall = (
                true_positives
                / (
                    true_positives
                    + false_negatives
                )
                if (
                    true_positives
                    + false_negatives
                ) > 0
                else 0.0
            )

            fbeta = (
                ModelEvaluation.calculate_fbeta(
                    precision,
                    recall,
                    beta
                )
            )

            scores.append(fbeta)

        if not scores:
            return 0.0

        return float(
            np.mean(scores)
        )

    @staticmethod
    def prepare_evaluation_data(
        feature_data,
        model,
        feature_columns
    ):

        data = feature_data.copy()

        X = data[
            feature_columns
        ].fillna(0)

        data["prediction_score"] = (
            model.predict_proba(X)[:, 1]
        )

        return data

    def find_best_threshold(
        self,
        feature_data,
        model,
        feature_columns
    ):

        logging.info(
            "Starting threshold tuning."
        )

        try:

            data = self.prepare_evaluation_data(
                feature_data,
                model,
                feature_columns
            )

            thresholds = np.arange(
                0.10,
                0.96,
                0.05
            )

            results = []

            for threshold in thresholds:

                fbeta = (
                    self.calculate_macro_fbeta(
                        data,
                        threshold,
                        beta=0.5
                    )
                )

                results.append(
                    {
                        "threshold": float(
                            threshold
                        ),
                        "macro_f0.5": float(
                            fbeta
                        )
                    }
                )

            results_df = pd.DataFrame(
                results
            )

            best_row = results_df.loc[
                results_df["macro_f0.5"].idxmax()
            ]

            best_threshold = float(
                best_row["threshold"]
            )

            best_score = float(
                best_row["macro_f0.5"]
            )

            logging.info(
                f"Best threshold: {best_threshold}"
            )

            logging.info(
                f"Best Macro F0.5: {best_score}"
            )

            return (
                best_threshold,
                best_score,
                results_df
            )

        except Exception as e:

            logging.error(
                "Error occurred during threshold tuning."
            )

            raise CustomException(
                e,
                sys
            )

    def evaluate(
        self,
        feature_data,
        model,
        feature_columns,
        threshold
    ):

        logging.info(
            "Starting model evaluation."
        )

        try:

            data = self.prepare_evaluation_data(
                feature_data,
                model,
                feature_columns
            )

            macro_fbeta = (
                self.calculate_macro_fbeta(
                    data,
                    threshold,
                    beta=0.5
                )
            )

            predicted_pairs = (
                data[
                    data["prediction_score"]
                    >= threshold
                ]
            )

            logging.info(
                f"Evaluation Macro F0.5: "
                f"{macro_fbeta:.4f}"
            )

            logging.info(
                f"Predicted candidate matches: "
                f"{len(predicted_pairs)}"
            )

            return macro_fbeta

        except Exception as e:

            logging.error(
                "Error occurred during model evaluation."
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":
    print("ModelEvaluation component loaded successfully.")