import os

class ConfigurationManager:
    def __init__(self):
        self.project_root = os.getcwd()

        self.dataset_dir = os.path.join(
            self.project_root,
            "dataset"
        )

        self.train_dir = os.path.join(
            self.dataset_dir,
            "train"
        )

        self.test_dir = os.path.join(
            self.dataset_dir,
            "test"
        )

        self.train_source1 = os.path.join(
            self.train_dir,
            "train_source1.tsv"
        )

        self.train_source2 = os.path.join(
            self.train_dir,
            "train_source2.tsv"
        )

        self.train_source3 = os.path.join(
            self.train_dir,
            "train_source3.tsv"
        )

        self.train_ground_truth = os.path.join(
            self.train_dir,
            "train_ground_truth.tsv"
        )

        self.test_source1 = os.path.join(
            self.test_dir,
            "test_source1.tsv"
        )

        self.test_source2 = os.path.join(
            self.test_dir,
            "test_source2.tsv"
        )

        self.test_source3 = os.path.join(
            self.test_dir,
            "test_source3.tsv"
        )

        self.artifacts_dir = os.path.join(
            self.project_root,
            "artifacts"
        )

        self.output_dir = os.path.join(
            self.project_root,
            "output"
        )