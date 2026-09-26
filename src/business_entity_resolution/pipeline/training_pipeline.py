import logging
import os
import sys
from business_entity_resolution.utils import logger
import pandas as pd
from sklearn.model_selection import train_test_split

from business_entity_resolution.components.data_ingestion import (
    DataIngestion
)

from business_entity_resolution.components.data_validation import (
    DataValidation
)

from business_entity_resolution.components.data_transformation import (
    DataTransformation
)

from business_entity_resolution.components.candidate_generation import (
    CandidateGeneration
)

from business_entity_resolution.components.feature_engineering import (
    FeatureEngineering
)

from business_entity_resolution.components.model_trainer import (
    ModelTrainer
)

from business_entity_resolution.components.model_evaluation import (
    ModelEvaluation
)

from business_entity_resolution.utils.exception import (
    CustomException
)


class TrainingPipeline:

    def __init__(
        self,
        artifacts_dir="artifacts"
    ):

        self.artifacts_dir = artifacts_dir

        os.makedirs(
            self.artifacts_dir,
            exist_ok=True
        )

    # =========================================================
    # GET TRUE MATCH IDS
    # =========================================================

    @staticmethod
    def get_matching_ids(
        ground_truth,
        source1_ids,
        source_prefix
    ):

        relevant_ground_truth = ground_truth[
            ground_truth["source1_entity_id"].isin(
                source1_ids
            )
        ].copy()

        matched_ids = (
            relevant_ground_truth[
                "matched_entity_ids"
            ]
            .dropna()
            .astype(str)
            .str.split(",")
            .explode()
            .str.strip()
        )

        matched_ids = matched_ids[
            matched_ids.str.startswith(
                source_prefix
            )
        ].unique()

        return matched_ids

    # =========================================================
    # CREATE SMALL TRAINING DATASET
    # =========================================================

    @staticmethod
    def prepare_small_dataset(
        source1,
        source2,
        source3,
        ground_truth,
        source1_limit,
        source2_limit,
        source3_limit
    ):

        logging.info(
            "Preparing smoke-test training dataset."
        )

        # -----------------------------------------------------
        # Select S1 entities that have known matches
        # -----------------------------------------------------

        matched_s1_ids = (
            ground_truth[
                ground_truth["matched_entity_ids"].notna()
                & ground_truth["matched_entity_ids"].ne("")
            ]["source1_entity_id"]
            .drop_duplicates()
        )

        sample_size = min(
            source1_limit,
            len(matched_s1_ids)
        )

        selected_s1_ids = (
            matched_s1_ids
            .sample(
                n=sample_size,
                random_state=42
            )
            .tolist()
        )

        # -----------------------------------------------------
        # Select S1 records
        # -----------------------------------------------------

        source1_small = source1[
            source1["entity_id"].isin(
                selected_s1_ids
            )
        ].copy()

        # -----------------------------------------------------
        # Select relevant ground truth
        # -----------------------------------------------------

        ground_truth_small = ground_truth[
            ground_truth["source1_entity_id"].isin(
                selected_s1_ids
            )
        ].copy()

        # -----------------------------------------------------
        # Find actual S2 matches
        # -----------------------------------------------------

        true_s2_ids = (
            TrainingPipeline.get_matching_ids(
                ground_truth_small,
                selected_s1_ids,
                "S2-"
            )
        )

        # -----------------------------------------------------
        # Find actual S3 matches
        # -----------------------------------------------------

        true_s3_ids = (
            TrainingPipeline.get_matching_ids(
                ground_truth_small,
                selected_s1_ids,
                "S3-"
            )
        )

        logging.info(
            f"True S2 records required: "
            f"{len(true_s2_ids)}"
        )

        logging.info(
            f"True S3 records required: "
            f"{len(true_s3_ids)}"
        )

        # -----------------------------------------------------
        # Keep true S2 records
        # -----------------------------------------------------

        source2_true = source2[
            source2["entity_id"].isin(
                true_s2_ids
            )
        ].copy()

        # -----------------------------------------------------
        # Keep true S3 records
        # -----------------------------------------------------

        source3_true = source3[
            source3["entity_id"].isin(
                true_s3_ids
            )
        ].copy()

        # -----------------------------------------------------
        # Add additional S2 records
        # -----------------------------------------------------

        remaining_s2 = source2[
            ~source2["entity_id"].isin(
                true_s2_ids
            )
        ]

        extra_s2_count = max(
            source2_limit - len(source2_true),
            0
        )

        source2_extra = remaining_s2.head(
            extra_s2_count
        )

        # -----------------------------------------------------
        # Add additional S3 records
        # -----------------------------------------------------

        remaining_s3 = source3[
            ~source3["entity_id"].isin(
                true_s3_ids
            )
        ]

        extra_s3_count = max(
            source3_limit - len(source3_true),
            0
        )

        source3_extra = remaining_s3.head(
            extra_s3_count
        )

        # -----------------------------------------------------
        # Combine S2
        # -----------------------------------------------------

        source2_small = pd.concat(
            [
                source2_true,
                source2_extra
            ],
            ignore_index=True
        )

        # -----------------------------------------------------
        # Combine S3
        # -----------------------------------------------------

        source3_small = pd.concat(
            [
                source3_true,
                source3_extra
            ],
            ignore_index=True
        )

        logging.info(
            f"Small S1 shape: {source1_small.shape}"
        )

        logging.info(
            f"Small S2 shape: {source2_small.shape}"
        )

        logging.info(
            f"Small S3 shape: {source3_small.shape}"
        )

        return (
            source1_small,
            source2_small,
            source3_small,
            ground_truth_small
        )

    # =========================================================
    # PREPARE FEATURES
    # =========================================================

    @staticmethod
    def prepare_features(
        candidates_s2,
        candidates_s3,
        feature_engineering
    ):

        features_s2 = (
            feature_engineering.create_features(
                candidates_s2
            )
        )

        features_s3 = (
            feature_engineering.create_features(
                candidates_s3
            )
        )

        features = pd.concat(
            [
                features_s2,
                features_s3
            ],
            ignore_index=True
        )

        return features

    # =========================================================
    # RUN PIPELINE
    # =========================================================

    def run(
        self,
        train_s1_limit=100,
        source2_limit=5000,
        source3_limit=5000
    ):

        logging.info(
            "Starting training pipeline."
        )

        try:

            # =================================================
            # 1. DATA INGESTION
            # =================================================

            logging.info(
                "Step 1: Data ingestion."
            )

            ingestion = DataIngestion()

            (
                source1,
                source2,
                source3,
                ground_truth
            ) = ingestion.initiate_data_ingestion()

            # =================================================
            # 2. SMALL DATASET FOR SMOKE TEST
            # =================================================

            logging.info(
                "Step 2: Preparing small dataset."
            )

            (
                source1,
                source2,
                source3,
                ground_truth
            ) = self.prepare_small_dataset(
                source1,
                source2,
                source3,
                ground_truth,
                train_s1_limit,
                source2_limit,
                source3_limit
            )

            print(
                f"S1 records: {len(source1)}"
            )

            print(
                f"S2 records: {len(source2)}"
            )

            print(
                f"S3 records: {len(source3)}"
            )

            # =================================================
            # 3. DATA VALIDATION
            # =================================================

            logging.info(
                "Step 3: Data validation."
            )

            validation = DataValidation(
                source1,
                source2,
                source3,
                ground_truth
            )

            validation.initiate_data_validation()

            # =================================================
            # 4. DATA TRANSFORMATION
            # =================================================

            logging.info(
                "Step 4: Data transformation."
            )

            transformer = DataTransformation()

            source1 = transformer.transform_source(
                source1
            )

            source2 = transformer.transform_source(
                source2
            )

            source3 = transformer.transform_source(
                source3
            )

            # =================================================
            # 5. TRAIN / VALIDATION SPLIT
            # =================================================

            logging.info(
                "Step 5: Train validation split."
            )

            source1_ids = (
                source1["entity_id"]
                .drop_duplicates()
            )

            train_ids, validation_ids = (
                train_test_split(
                    source1_ids,
                    test_size=0.20,
                    random_state=42
                )
            )

            train_source1 = source1[
                source1["entity_id"].isin(
                    train_ids
                )
            ].copy()

            validation_source1 = source1[
                source1["entity_id"].isin(
                    validation_ids
                )
            ].copy()

            train_ground_truth = ground_truth[
                ground_truth[
                    "source1_entity_id"
                ].isin(
                    train_ids
                )
            ].copy()

            validation_ground_truth = ground_truth[
                ground_truth[
                    "source1_entity_id"
                ].isin(
                    validation_ids
                )
            ].copy()

            print(
                f"Train S1 records: "
                f"{len(train_source1)}"
            )

            print(
                f"Validation S1 records: "
                f"{len(validation_source1)}"
            )

            # =================================================
            # 6. CANDIDATE GENERATION
            # =================================================

            logging.info(
                "Step 6: Candidate generation."
            )

            candidate_generator = (
                CandidateGeneration()
            )

            (
                train_candidates_s2,
                train_candidates_s3
            ) = (
                candidate_generator.generate_candidates(
                    train_source1,
                    source2,
                    source3
                )
            )

            (
                validation_candidates_s2,
                validation_candidates_s3
            ) = (
                candidate_generator.generate_candidates(
                    validation_source1,
                    source2,
                    source3
                )
            )

            print(
                f"Train S2 candidates: "
                f"{len(train_candidates_s2)}"
            )

            print(
                f"Train S3 candidates: "
                f"{len(train_candidates_s3)}"
            )

            print(
                f"Validation S2 candidates: "
                f"{len(validation_candidates_s2)}"
            )

            print(
                f"Validation S3 candidates: "
                f"{len(validation_candidates_s3)}"
            )

            # =================================================
            # 7. FEATURE ENGINEERING
            # =================================================

            logging.info(
                "Step 7: Feature engineering."
            )

            feature_engineering = (
                FeatureEngineering()
            )

            train_features = (
                self.prepare_features(
                    train_candidates_s2,
                    train_candidates_s3,
                    feature_engineering
                )
            )

            validation_features = (
                self.prepare_features(
                    validation_candidates_s2,
                    validation_candidates_s3,
                    feature_engineering
                )
            )

            print(
                f"Train feature rows: "
                f"{len(train_features)}"
            )

            print(
                f"Validation feature rows: "
                f"{len(validation_features)}"
            )

            # =================================================
            # 8. MODEL TRAINING
            # =================================================

            logging.info(
                "Step 8: Model training."
            )

            model_trainer = ModelTrainer(
                artifacts_dir=self.artifacts_dir
            )

            (
                model,
                feature_columns
            ) = model_trainer.train_model(
                train_features,
                train_ground_truth
            )

            # =================================================
            # 9. THRESHOLD TUNING
            # =================================================

            logging.info(
                "Step 9: Threshold tuning."
            )

            evaluator = ModelEvaluation()

            validation_features_labeled = (
                ModelTrainer.create_labels(
                    validation_features,
                    validation_ground_truth
                )
            )

            (
                best_threshold,
                best_score,
                threshold_results
            ) = evaluator.find_best_threshold(
                validation_features_labeled,
                model,
                feature_columns
            )

            # =================================================
            # 10. SAVE BEST THRESHOLD
            # =================================================

            threshold_path = os.path.join(
                self.artifacts_dir,
                "best_threshold.txt"
            )

            with open(
                threshold_path,
                "w"
            ) as file:

                file.write(
                    str(best_threshold)
                )

            # =================================================
            # 11. SAVE THRESHOLD RESULTS
            # =================================================

            threshold_results_path = os.path.join(
                self.artifacts_dir,
                "threshold_results.tsv"
            )

            threshold_results.to_csv(
                threshold_results_path,
                sep="\t",
                index=False
            )

            # =================================================
            # 12. FINAL VALIDATION
            # =================================================

            logging.info(
                "Step 10: Final validation."
            )

            validation_score = evaluator.evaluate(
                validation_features_labeled,
                model,
                feature_columns,
                best_threshold
            )

            print(
                "\n===================================="
            )

            print(
                "TRAINING COMPLETED"
            )

            print(
                "===================================="
            )

            print(
                f"Best threshold: "
                f"{best_threshold:.4f}"
            )

            print(
                f"Validation Macro F0.5: "
                f"{validation_score:.4f}"
            )

            print(
                f"Model saved at:"
            )

            print(
                model_trainer.model_path
            )

            print(
                f"Threshold saved at:"
            )

            print(
                threshold_path
            )

            print(
                "===================================="
            )

            return {
                "model": model,
                "feature_columns": feature_columns,
                "threshold": best_threshold,
                "validation_score": validation_score
            }

        except Exception as e:

            logging.error(
                "Error occurred during training pipeline."
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    pipeline = TrainingPipeline()

    pipeline.run(
        train_s1_limit=100,
        source2_limit=5000,
        source3_limit=5000
    )