import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np


class CrossValidationDataset:
    def __init__(
        self, seed, root_path="./dataset", save_dir="CrossValidation", verbose=False
    ):
        # ---- setting ----
        self.root_path = root_path
        self.save_dir = save_dir
        self.verbose = verbose
        self.seed = seed
        random.seed(seed)

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
        assert len(self.not_sekkai) + len(self.sekkai) == len(
            self.all
        ), f"len(not_sekkai) + len(sekkai) != len(all) {len(self.not_sekkai)} + {len(self.sekkai)} != {len(self.all)}"

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

    def __call__(self, index):
        self.index = index

        # ---- shuffle ----
        random.seed(self.seed)
        random.shuffle(self.sekkai)
        random.seed(self.seed)
        random.shuffle(self.not_sekkai)

        # ---- split ----
        train_sekkai = self.sekkai[: self.sekkai_train_size]
        val_sekkai = self.sekkai[
            self.sekkai_train_size : self.sekkai_train_size + self.sekkai_val_size
        ]
        test_sekkai = self.sekkai[self.sekkai_train_size + self.sekkai_val_size :]

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
            f"./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/images",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/masks",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/{self.save_dir}/cv{self.index}_sekkai_{split}Dataset/images",
            exist_ok=True,
        )
        os.makedirs(
            f"./dataset/{self.save_dir}/cv{self.index}_sekkai_{split}Dataset/masks",
            exist_ok=True,
        )

        # sekkai_dataset
        print(f">>> sekkai_{split}Dataset")
        for se in sekkai:
            # ---- copy sekkai ----
            os.system(
                f"cp {se} ./dataset/{self.save_dir}/cv{self.index}_sekkai_{split}Dataset/images"
            )
            os.system(
                f"cp {str(se).replace('images', 'masks')} ./dataset/{self.save_dir}/cv{self.index}_sekkai_{split}Dataset/masks"
            )

        # dataset
        print(f">>> {split}Dataset")
        for se in sekkai:
            # ---- copy sekkai ----
            os.makedirs(
                f"./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/images",
                exist_ok=True,
            )
            os.system(
                f"cp {se} ./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/images"
            )
            os.system(
                f"cp {str(se).replace('images', 'masks')} ./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/masks"
            )
        for ns in not_sekkai:
            # ---- copy not sekkai ----
            os.makedirs(
                f"./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/images",
                exist_ok=True,
            )
            os.system(
                f"cp {ns} ./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/images"
            )
            os.system(
                f"cp {str(ns).replace('images', 'masks')} ./dataset/{self.save_dir}/cv{self.index}_{split}Dataset/masks"
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


class CrossValidationDatasetOriginal:
    def __init__(self, verbose=False):
        # ---- setting ----
        self.root_path = Path("dataset/CrossValidation")
        self.save_dir = Path("dataset/CrossValidation_original")
        self.verbose = verbose

        self.target_path = Path("./dataset/original_dataset_only_left")
        target_train_path = self.target_path / "TrainDataset" / "images"
        target_val_path = self.target_path / "ValDataset" / "images"
        target_test_path = self.target_path / "TestDataset" / "images"
        self.target_list = (
            list(target_train_path.glob("*.png"))
            + list(target_val_path.glob("*.png"))
            + list(target_test_path.glob("*.png"))
        )

        os.makedirs(self.save_dir, exist_ok=True)

        if verbose:
            self._verbose()

    def _verbose(self):
        print(f">>> Target: {len(self.target_list)}")

    def _makedirs(self):
        os.makedirs(
            self.save_dir / f"cv{self.index}_TrainDataset/images", exist_ok=True
        )
        os.makedirs(self.save_dir / f"cv{self.index}_TrainDataset/masks", exist_ok=True)
        os.makedirs(self.save_dir / f"cv{self.index}_ValDataset/images", exist_ok=True)
        os.makedirs(self.save_dir / f"cv{self.index}_ValDataset/masks", exist_ok=True)
        os.makedirs(self.save_dir / f"cv{self.index}_TestDataset/images", exist_ok=True)
        os.makedirs(self.save_dir / f"cv{self.index}_TestDataset/masks", exist_ok=True)
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_TrainDataset/images", exist_ok=True
        )
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_TrainDataset/masks", exist_ok=True
        )
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_ValDataset/images", exist_ok=True
        )
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_ValDataset/masks", exist_ok=True
        )
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_TestDataset/images", exist_ok=True
        )
        os.makedirs(
            self.save_dir / f"cv{self.index}_sekkai_TestDataset/masks", exist_ok=True
        )

    def __call__(self, index):
        self.index = index
        self._makedirs()

        # ---- get path ----
        train_path = self.root_path / f"cv{self.index}_TrainDataset"
        val_path = self.root_path / f"cv{self.index}_ValDataset"
        test_path = self.root_path / f"cv{self.index}_TestDataset"

        sekkai_train_path = self.root_path / f"cv{self.index}_sekkai_TrainDataset"
        sekkai_val_path = self.root_path / f"cv{self.index}_sekkai_ValDataset"
        sekkai_test_path = self.root_path / f"cv{self.index}_sekkai_TestDataset"

        # ---- get file list ----
        train_list = list(train_path.glob("images/*.png"))
        val_list = list(val_path.glob("images/*.png"))
        test_list = list(test_path.glob("images/*.png"))

        train_filename = [f.stem for f in train_list]
        val_filename = [f.stem for f in val_list]
        test_filename = [f.stem for f in test_list]

        sekkai_train_list = list(sekkai_train_path.glob("images/*.png"))
        sekkai_val_list = list(sekkai_val_path.glob("images/*.png"))
        sekkai_test_list = list(sekkai_test_path.glob("images/*.png"))

        if self.verbose:
            print(f">>> Train: {len(train_list)}")
            print(f">>> Val: {len(val_list)}")
            print(f">>> Test: {len(test_list)}")
            print(f">>> Sekkai Train: {len(sekkai_train_list)}")
            print(f">>> Sekkai Val: {len(sekkai_val_list)}")
            print(f">>> Sekkai Test: {len(sekkai_test_list)}")

        sekkai_train_filename = [f.stem for f in sekkai_train_list]
        sekkai_val_filename = [f.stem for f in sekkai_val_list]
        sekkai_test_filename = [f.stem for f in sekkai_test_list]

        for filename in train_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_TrainDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )
        for filename in val_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_ValDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )
        for filename in test_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_TestDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )

        for filename in sekkai_train_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_sekkai_TrainDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )
        for filename in sekkai_val_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_sekkai_ValDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )
        for filename in sekkai_test_filename:
            if filename in list(map(lambda x: x.stem, self.target_list)):
                target_file_path = [f for f in self.target_list if f.stem == filename][
                    0
                ]
                save_file_path = (
                    self.save_dir
                    / f"cv{self.index}_sekkai_TestDataset"
                    / "images"
                    / f"{filename}.png"
                )
                os.system(f"cp {target_file_path} {save_file_path}")
                os.system(
                    f"cp {str(target_file_path).replace('images', 'masks')} {str(save_file_path).replace('images', 'masks')}"
                )


class CropConstParamD:
    def __init__(
        self,
        root_dir: str = "CrossValidation_original",
        mode: str = "cv",
        param_d: float = 0.625,
        verbose: bool = False,
        image_size: int = 352,
    ) -> None:

        self.root_path = Path(f"dataset/{root_dir}")
        self.save_dir = Path(f"dataset/CrossValidation_original_crop_{param_d}")
        self.mode = mode
        self.param_d = param_d
        self.verbose = verbose
        self.image_size = image_size

        os.makedirs(self.save_dir, exist_ok=True)

        self.label_count = 0

    def _get_dataset_list_cv(self) -> None:
        self.train_list = list(
            self.root_path.glob(f"cv{self.index}_TrainDataset/images/*.png")
        )
        self.val_list = list(
            self.root_path.glob(f"cv{self.index}_ValDataset/images/*.png")
        )
        self.test_list = list(
            self.root_path.glob(f"cv{self.index}_TestDataset/images/*.png")
        )

        self.sekkai_train_list = list(
            self.root_path.glob(f"cv{self.index}_sekkai_TrainDataset/images/*.png")
        )
        self.sekkai_val_list = list(
            self.root_path.glob(f"cv{self.index}_sekkai_ValDataset/images/*.png")
        )
        self.sekkai_test_list = list(
            self.root_path.glob(f"cv{self.index}_sekkai_TestDataset/images/*.png")
        )

    def _crop_image(self, image_path: Path, mask_path: Path) -> None:
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        # resize image
        image = cv2.resize(image, (1430, 384))
        mask = cv2.resize(mask, (1430, 384))
        height, width = image.shape[:2]

        # w-0.375(w-h) = 0.625w + 0.375h
        crop_x = int(self.param_d * width + (1 - self.param_d) * height)
        crop_image = image[:, :crop_x]
        crop_mask = mask[:, :crop_x]

        # resize original size
        crop_image = cv2.resize(crop_image, (self.image_size, self.image_size))
        crop_mask = cv2.resize(crop_mask, (self.image_size, self.image_size))
        self.label_count += 1 if np.sum(crop_mask) > 0 else 0

        cv2.imwrite(str(self.save_dir_dataset / image_path.name), crop_image)
        cv2.imwrite(
            str(self.save_dir_dataset / mask_path.name).replace("images", "masks"),
            crop_mask,
        )

        if self.verbose:
            print(f">>> {str(self.save_dir_dataset / image_path.name)}")
            print(
                f">>> {str(self.save_dir_dataset / image_path.name).replace('images', 'masks')}"
            )
            print("=" * 10)

    def _make_crop_dataset(self, data_list) -> None:
        for image_path in data_list:
            image_path = Path(image_path)
            mask_path = Path(str(image_path).replace("images", "masks"))
            self._crop_image(image_path, mask_path)

    def __call__(self) -> None:
        if self.mode == "cv":
            for index in range(5):
                self.index = index

                self._get_dataset_list_cv()
                for data_list in [
                    self.train_list,
                    self.val_list,
                    self.test_list,
                    self.sekkai_train_list,
                    self.sekkai_val_list,
                    self.sekkai_test_list,
                ]:

                    self.save_dir_dataset = (
                        self.save_dir
                        / data_list[0].parents[1].name
                        / data_list[0].parents[0].name
                    )
                    print(f">>> save_dir: {self.save_dir_dataset}")
                    os.makedirs(self.save_dir_dataset, exist_ok=True)
                    os.makedirs(
                        str(self.save_dir_dataset).replace("images", "masks"),
                        exist_ok=True,
                    )
                    self._make_crop_dataset(data_list)
                    print(f">>> label_count: {self.label_count}")
                    self.label_count = 0


class CrossValidationDatasetMAE:
    def __init__(
        self,
        root_dir: str = "CrossValidation",
        save_dir: str = "mae/dataset/TrainDataset/images",
    ):
        self.root_path = Path(f"dataset/{root_dir}")
        self.mae_path = Path("dataset/mae/dataset/TrainDataset/images/")
        self.save_dir = Path(f"dataset/{save_dir}")

    def _get_dataset_list_cv(self, index) -> None:
        self.train_list = list(
            self.root_path.glob(f"cv{index}_TrainDataset/images/*.png")
        )

    def _get_mae_dataset(self, index: int = 0) -> tuple[list[Path], list[str]]:

        mae_list = list(self.mae_path.glob(f"{index}/*.png"))
        assert len(mae_list) == 1141, f"mae dataset is not 1141, but {len(mae_list)}"

        train_list = list(self.root_path.glob(f"cv{index}_TrainDataset/images/*.png"))
        assert (
            len(train_list) == 831
        ), f"train dataset is not 831, but {len(train_list)}"

        mae_filename_list = [f.stem for f in mae_list]
        train_filename_list = [f.stem for f in train_list]

        not_train_file_name_list = list(
            set(mae_filename_list) - set(train_filename_list)
        )
        assert (
            len(not_train_file_name_list) == 310
        ), f"not mae dataset is not 310, but {len(not_train_file_name_list)}"

        return mae_list, not_train_file_name_list

    def _make_mae_dataset(self, index, data_list, not_train_file_name_list) -> None:
        os.makedirs(self.save_dir / f"{index}", exist_ok=True)
        for image_path in data_list:
            image_path = Path(image_path)

            save_file_path = self.save_dir / f"{index}" / f"{image_path.stem}.png"
            os.system(f"cp {image_path} {save_file_path}")

        for file_name in not_train_file_name_list:
            save_file_path = self.save_dir / f"{index}" / f"{file_name}.png"
            os.system(f"cp {self.mae_path / '0' / f'{file_name}.png'} {save_file_path}")

    def __call__(self) -> None:
        _, not_train_file_name_list = self._get_mae_dataset(0)

        for index in range(1, 5):
            self._get_dataset_list_cv(index)
            self._make_mae_dataset(index, self.train_list, not_train_file_name_list)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_type",
        type=str,
        default="default",
        help="default, original, crop",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    data_type = args.data_type

    if data_type == "default":
        for index, seed in enumerate([31, 42, 71, 91], start=1):
            cv = CrossValidationDataset(
                verbose=True,
                seed=seed,
            )
            cv(index=index)

            cv_original = CrossValidationDatasetOriginal(verbose=True)
            for index in range(0, 5):
                print(f">>> cv{index}")
                cv_original(index=index)
                print("-" * 50)

    elif data_type == "mae":
        mae = CrossValidationDatasetMAE()
        mae()

    elif data_type == "original":
        for index, seed in enumerate([31, 42, 71, 91], start=1):
            cv = CrossValidationDataset(
                root_path="dataset/original_dataset_only_left",
                save_dir="CrossValidation_original",
                verbose=True,
                seed=seed,
            )
            cv(index=index)

            cv_original = CrossValidationDatasetOriginal(verbose=True)
            for index in range(0, 5):
                print(f">>> cv{index}")
                cv_original(index=index)
                print("-" * 50)

    elif data_type == "crop":
        param_d_list = [
            0.625,
            0.615,
            0.620,
            0.630,
            0.635,
            0.550,
            0.575,
            0.600,
            0.650,
            0.675,
            0.700,
        ]
        for param_d in param_d_list:
            print(f">>> param_d: {param_d}")
            crop = CropConstParamD(param_d=param_d)
            crop()
            print("-" * 50)
