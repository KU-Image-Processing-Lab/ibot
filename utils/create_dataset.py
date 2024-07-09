import os
import random
from pathlib import Path


class CrossValidationDataset:
    def __init__(self, root_path="./dataset", verbose=False):
        self.root_path = root_path
        self.verbose = verbose

        # ---- sekkai dataset ----
        self.sekkai_train_list = self.get_data_set("Train", sekkai=True)
        self.sekkai_val_list = self.get_data_set("Val", sekkai=True)
        self.sekkai_test_list = self.get_data_set("Test", sekkai=True)
        self.sekkai = list(
            self.sekkai_train_list | self.sekkai_val_list | self.sekkai_test_list
        )

        # ---- sekkai and not_sekkai dataset ----
        self.train_list = self.get_data_set("Train", sekkai=False)
        self.val_list = self.get_data_set("Val", sekkai=False)
        self.test_list = self.get_data_set("Test", sekkai=False)
        self.all = list(self.train_list | self.val_list | self.test_list)

        # ---- not sekkai dataset ----
        self.not_sekkai = []
        for al in self.all:
            # ---- file name ----
            file_name = al.stem
            if not any(file_name == f.stem for f in self.sekkai):
                self.not_sekkai.append(al)
        assert len(self.not_sekkai) + len(self.sekkai) == len(self.all), \
            f"len(not_sekkai) + len(sekkai) != len(all) {len(self.not_sekkai)} + {len(self.sekkai)} != {len(self.all)}"

        # ---- split size ----
        self._get_split_size()

        if verbose:
            self._verbose()

    def get_data_set(self, phase, sekkai: bool):
        if sekkai:
            return set(
                Path(f"./{self.root_path}/sekkai_{phase}Dataset/images").glob("*.png")
            )
        return set(Path(f"./{self.root_path}/{phase}Dataset/images").glob("*.png"))

    def __call__(self, seed):
        # ---- set random seed ----
        self.seed = seed
        random.seed(self.seed)

        # ---- shuffle ----
        random.shuffle(self.sekkai)
        random.shuffle(self.not_sekkai)

        # ---- split ----
        train_sekkai = self.sekkai[: self.sekkai_train_size]
        val_sekkai = self.sekkai[
            self.sekkai_train_size : self.sekkai_train_size + self.sekkai_val_size
        ]
        test_sekkai = self.sekkai[
            self.sekkai_train_size + self.sekkai_val_size :
        ]

        train_not_sekkai = self.not_sekkai[: self.not_sekkai_train_size]
        val_not_sekkai = self.not_sekkai[
            self.not_sekkai_train_size : self.not_sekkai_train_size
            + self.not_sekkai_val_size
        ]
        test_not_sekkai = self.not_sekkai[
            self.not_sekkai_train_size + self.not_sekkai_val_size :
        ]

        # ---- save ----
        self._save(train_sekkai, train_not_sekkai, "Train")
        self._save(val_sekkai, val_not_sekkai, "Val")
        self._save(test_sekkai, test_not_sekkai, "Test")

    def _save(self, sekkai, not_sekkai, split):
        os.makedirs(
            f"./dataset/CrossValidation/seed{self.seed}_{split}Dataset/images",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/CrossValidation/seed{self.seed}_{split}Dataset/masks",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/CrossValidation/seed{self.seed}_sekkai_{split}Dataset/images",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/CrossValidation/seed{self.seed}_sekkai_{split}Dataset/masks",
            exist_ok=True,
        )

        # sekkai_dataset
        print(f">>> sekkai_{split}Dataset")
        for se in sekkai:
            # ---- copy sekkai ----
            os.system(
                f"cp {se} ./dataset/CrossValidation/seed{self.seed}_sekkai_{split}Dataset/images"
            )
            os.system(
                f"cp {str(se).replace('images', 'masks')} ./dataset/CrossValidation/seed{self.seed}_sekkai_{split}Dataset/masks"
            )

        # dataset
        print(f">>> {split}Dataset")
        for se in sekkai:
            # ---- copy sekkai ----
            os.makedirs(
                f"./dataset/CrossValidation/seed{self.seed}_{split}Dataset/images",
                exist_ok=True,
            )
            os.system(
                f"cp {se} ./dataset/CrossValidation/seed{self.seed}_{split}Dataset/images"
            )
            os.system(
                f"cp {str(se).replace('images', 'masks')} ./dataset/CrossValidation/seed{self.seed}_{split}Dataset/masks"
            )
        for ns in not_sekkai:
            # ---- copy not sekkai ----
            os.makedirs(
                f"./dataset/CrossValidation/seed{self.seed}_{split}Dataset/images",
                exist_ok=True,
            )
            os.system(
                f"cp {ns} ./dataset/CrossValidation/seed{self.seed}_{split}Dataset/images"
            )
            os.system(
                f"cp {str(ns).replace('images', 'masks')} ./dataset/CrossValidation/seed{self.seed}_{split}Dataset/masks"
            )

    def _get_split_size(self):
        self.sekkai_train_size = len(self.sekkai_train_list)
        self.sekkai_val_size = len(self.sekkai_val_list)
        self.sekkai_test_size = len(self.sekkai_test_list)

        self.not_sekkai_train_size = len(self.train_list) - len(self.sekkai_train_list)
        self.not_sekkai_val_size = len(self.val_list) - len(self.sekkai_val_list)
        self.not_sekkai_test_size = len(self.test_list) - len(self.sekkai_test_list)

        self.sekkai_size = len(self.sekkai)
        self.not_sekkai_size = len(self.not_sekkai)

    def _verbose(self):
        print(f">>> Sekkai Train: {len(self.sekkai_train_list)}")
        print(f">>> Sekkai Val: {len(self.sekkai_val_list)}")
        print(f">>> Sekkai Test: {len(self.sekkai_test_list)}")

        print(f">>> Not Sekkai Train: {self.not_sekkai_train_size}")
        print(f">>> Not Sekkai Val: {self.not_sekkai_val_size}")
        print(f">>> Not Sekkai Test: {self.not_sekkai_test_size}")

        print(f">>> All: {len(self.all)}")
        print(f">>> Sekkai: {len(self.sekkai)}")
        print(f">>> Not Sekkai: {len(self.not_sekkai)}")


if __name__ == "__main__":
    for i in [31, 42, 71, 91, 101]:
        cv = CrossValidationDataset(verbose=True)
        cv(i)
