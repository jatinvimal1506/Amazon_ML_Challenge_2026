import logging
import sys

import pandas as pd

from business_entity_resolution.utils.exception import (
    CustomException
)


class CandidateGeneration:

    def __init__(self):
        pass

    @staticmethod
    def create_blocking_keys(df):
        df = df.copy()

        # Name-based blocking key
        df["name_block"] = (
            df["business_name_clean"]
            .fillna("")
            .astype(str)
            .str[:4]
        )

        df["name_blocking_key"] = (
            df["country"].fillna("").astype(str)
            + "_"
            + df["name_block"]
        )

        # Address-based blocking key
        df["address_block"] = (
            df["business_address_clean"]
            .fillna("")
            .astype(str)
            .str[:6]
        )

        df["address_blocking_key"] = (
            df["country"].fillna("").astype(str)
            + "_"
            + df["address_block"]
        )

        return df

    @staticmethod
    def generate_candidates_from_key(
        source1,
        source2,
        blocking_key,
        source_suffix
    ):

        # Do not generate candidates for empty blocking keys.
        source1_blocked = source1[
            source1[blocking_key].ne("_")
            & source1[blocking_key].ne("")
        ].copy()

        source2_blocked = source2[
            source2[blocking_key].ne("_")
            & source2[blocking_key].ne("")
        ].copy()

        candidates = source1_blocked.merge(
            source2_blocked,
            on=blocking_key,
            how="inner",
            suffixes=(
                "_s1",
                f"_{source_suffix}"
            )
        )

        return candidates

    @staticmethod
    def remove_duplicate_pairs(
        candidates,
        source_id_column
    ):

        pair_columns = [
            "entity_id_s1",
            source_id_column
        ]

        candidates = candidates.drop_duplicates(
            subset=pair_columns
        )

        return candidates

    def generate_candidates(
        self,
        source1,
        source2,
        source3
    ):

        logging.info(
            "Starting candidate generation."
        )

        try:
            # Create blocking keys
            source1 = self.create_blocking_keys(
                source1
            )

            source2 = self.create_blocking_keys(
                source2
            )

            source3 = self.create_blocking_keys(
                source3
            )

            # S1 -> S2
            # Name blocking
            candidates_s2_name = (
                self.generate_candidates_from_key(
                    source1,
                    source2,
                    "name_blocking_key",
                    "s2"
                )
            )

            # Address blocking
            candidates_s2_address = (
                self.generate_candidates_from_key(
                    source1,
                    source2,
                    "address_blocking_key",
                    "s2"
                )
            )

            # Combine both blocking strategies
            candidates_s2 = pd.concat(
                [
                    candidates_s2_name,
                    candidates_s2_address
                ],
                ignore_index=True
            )

            candidates_s2 = (
                self.remove_duplicate_pairs(
                    candidates_s2,
                    "entity_id_s2"
                )
            )

            # S1 -> S3
            # Name blocking
            candidates_s3_name = (
                self.generate_candidates_from_key(
                    source1,
                    source3,
                    "name_blocking_key",
                    "s3"
                )
            )

            # Address blocking
            candidates_s3_address = (
                self.generate_candidates_from_key(
                    source1,
                    source3,
                    "address_blocking_key",
                    "s3"
                )
            )

            # Combine both blocking strategies
            candidates_s3 = pd.concat(
                [
                    candidates_s3_name,
                    candidates_s3_address
                ],
                ignore_index=True
            )

            candidates_s3 = (
                self.remove_duplicate_pairs(
                    candidates_s3,
                    "entity_id_s3"
                )
            )

            logging.info(
                f"S2 candidate pairs: {len(candidates_s2)}"
            )

            logging.info(
                f"S3 candidate pairs: {len(candidates_s3)}"
            )

            logging.info(
                "Candidate generation completed successfully."
            )

            return (
                candidates_s2,
                candidates_s3
            )

        except Exception as e:

            logging.error(
                "Error occurred during candidate generation."
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":
    print("CandidateGeneration component loaded successfully.")