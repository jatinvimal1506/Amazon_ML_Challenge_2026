import logging
import sys
import re
import pandas as pd

from business_entity_resolution.utils.exception import (
    CustomException
)


class CandidateGeneration:
    def __init__(self):
        pass

    @staticmethod
    def create_numeric_address_keys(df):

        numeric_keys = []

        for address, country in zip(
            df["business_address_clean"],
            df["country"]
        ):

            keys = []

            if pd.notna(address):

                numbers = re.findall(
                    r"\b\d{3,}\b",
                    str(address)
                )

                for number in numbers:

                    if pd.notna(country):

                        normalized_number = (
                            number.lstrip("0")
                            or "0"
                        )

                        key = (
                            str(country).lower()
                            + "_"
                            + normalized_number
                        )

                        keys.append(key)

            numeric_keys.append(keys)

        return numeric_keys

    @staticmethod
    def create_blocking_keys(df):

        df = df.copy()

        # -----------------------------
        # Name-based blocking
        # -----------------------------

        df["name_block"] = (
            df["business_name_clean"]
            .fillna("")
            .astype(str)
            .str[:4]
        )

        df["name_blocking_key"] = (
            df["country"]
            .fillna("")
            .astype(str)
            .str.lower()
            + "_"
            + df["name_block"]
        )

        # -----------------------------
        # Address-based blocking
        # -----------------------------

        df["address_block"] = (
            df["business_address_clean"]
            .fillna("")
            .astype(str)
            .str[:6]
        )

        df["address_blocking_key"] = (
            df["country"]
            .fillna("")
            .astype(str)
            .str.lower()
            + "_"
            + df["address_block"]
        )

        # -----------------------------
        # Numeric address blocking
        # -----------------------------

        df["numeric_address_blocking_keys"] = (
            CandidateGeneration.create_numeric_address_keys(
                df
            )
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
    def generate_candidates_from_numeric_key(
        source1,
        source2,
        source_suffix
    ):

        # Keep only records that have at least one
        # usable numeric address key.
        source1_numeric = source1[
            source1["numeric_address_blocking_keys"].map(
                len
            ) > 0
        ].copy()

        source2_numeric = source2[
            source2["numeric_address_blocking_keys"].map(
                len
            ) > 0
        ].copy()

        if source1_numeric.empty or source2_numeric.empty:

            return pd.DataFrame()

        # One row per numeric blocking key.
        source1_numeric = (
            source1_numeric
            .explode(
                "numeric_address_blocking_keys"
            )
        )

        source2_numeric = (
            source2_numeric
            .explode(
                "numeric_address_blocking_keys"
            )
        )

        candidates = source1_numeric.merge(
            source2_numeric,
            on="numeric_address_blocking_keys",
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

            # -----------------------------
            # Create blocking keys
            # -----------------------------

            source1 = self.create_blocking_keys(
                source1
            )

            source2 = self.create_blocking_keys(
                source2
            )

            source3 = self.create_blocking_keys(
                source3
            )

            # ==================================================
            # S1 -> S2
            # ==================================================

            # Name blocking
            candidates_s2_name = (
                self.generate_candidates_from_key(
                    source1,
                    source2,
                    "name_blocking_key",
                    "s2"
                )
            )

            # Address prefix blocking
            candidates_s2_address = (
                self.generate_candidates_from_key(
                    source1,
                    source2,
                    "address_blocking_key",
                    "s2"
                )
            )

            # Numeric address blocking
            candidates_s2_numeric = (
                self.generate_candidates_from_numeric_key(
                    source1,
                    source2,
                    "s2"
                )
            )

            logging.info(
                f"S2 name candidates: "
                f"{len(candidates_s2_name)}"
            )

            logging.info(
                f"S2 address candidates: "
                f"{len(candidates_s2_address)}"
            )

            logging.info(
                f"S2 numeric address candidates: "
                f"{len(candidates_s2_numeric)}"
            )

            # Combine all S2 blocking strategies
            candidates_s2 = pd.concat(
                [
                    candidates_s2_name,
                    candidates_s2_address,
                    candidates_s2_numeric
                ],
                ignore_index=True
            )

            candidates_s2 = (
                self.remove_duplicate_pairs(
                    candidates_s2,
                    "entity_id_s2"
                )
            )

            # ==================================================
            # S1 -> S3
            # ==================================================

            # Name blocking
            candidates_s3_name = (
                self.generate_candidates_from_key(
                    source1,
                    source3,
                    "name_blocking_key",
                    "s3"
                )
            )

            # Address prefix blocking
            candidates_s3_address = (
                self.generate_candidates_from_key(
                    source1,
                    source3,
                    "address_blocking_key",
                    "s3"
                )
            )

            # Numeric address blocking
            candidates_s3_numeric = (
                self.generate_candidates_from_numeric_key(
                    source1,
                    source3,
                    "s3"
                )
            )

            logging.info(
                f"S3 name candidates: "
                f"{len(candidates_s3_name)}"
            )

            logging.info(
                f"S3 address candidates: "
                f"{len(candidates_s3_address)}"
            )

            logging.info(
                f"S3 numeric address candidates: "
                f"{len(candidates_s3_numeric)}"
            )

            # Combine all S3 blocking strategies
            candidates_s3 = pd.concat(
                [
                    candidates_s3_name,
                    candidates_s3_address,
                    candidates_s3_numeric
                ],
                ignore_index=True
            )

            candidates_s3 = (
                self.remove_duplicate_pairs(
                    candidates_s3,
                    "entity_id_s3"
                )
            )

            # -----------------------------
            # Final logging
            # -----------------------------

            logging.info(
                f"S2 candidate pairs: "
                f"{len(candidates_s2)}"
            )

            logging.info(
                f"S3 candidate pairs: "
                f"{len(candidates_s3)}"
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