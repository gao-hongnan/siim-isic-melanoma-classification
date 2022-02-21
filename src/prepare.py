import os
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from config import global_params

from src import dataset, make_folds, transformation, utils


def return_filepath(
    image_id: str,
    folder: Path,
    extension: str,
) -> str:
    """Add a new column image_path to the train and test csv.
    We can call the images easily in __getitem__ in Dataset.

    If the image_id has extension already, then there is no need to add the extension.

    Args:
        image_id (str): The unique image id: 1000015157.jpg
        folder (Path, optional): The train folder. Defaults to FILES().train_images.
        extension (str, optional): The extension of the image. Defaults to ".jpg".

    Returns:
        image_path (str): The path to the image: "c:\\users\\reighns\\kaggle_projects\\cassava\\data\\train\\1000015157.jpg"
    """
    # TODO: Consider using Path instead os for consistency.
    image_path = os.path.join(folder, f"{image_id}{extension}")
    return image_path


def prepare_data(pipeline_config: global_params.PipelineConfig) -> pd.DataFrame:
    """Call a sequential number of steps to prepare the data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_col_name (str): The column name of the unique image id. In Cassava, it is "image_id".

    Returns:
        df_train (pd.DataFrame): The train dataframe.
        df_test (pd.DataFrame): The test dataframe.
        df_folds (pd.DataFrame): The folds dataframe with an additional column "fold".
        df_sub (pd.DataFrame): The submission dataframe.
    """

    df_train = pd.read_csv(pipeline_config.files.train_csv)
    df_test = pd.read_csv(pipeline_config.files.test_csv)
    df_sub = pd.read_csv(pipeline_config.files.sub_csv)

    df_train["image_path"] = df_train[
        pipeline_config.folds.image_col_name
    ].apply(
        lambda x: return_filepath(
            image_id=x,
            folder=pipeline_config.files.train_images,
            extension=pipeline_config.folds.image_extension,
        )
    )
    df_test["image_path"] = df_test[pipeline_config.folds.image_col_name].apply(
        lambda x: return_filepath(
            x,
            folder=pipeline_config.files.test_images,
            extension=pipeline_config.folds.image_extension,
        )
    )

    df_folds = make_folds.make_folds(
        train_csv=df_train, pipeline_config=pipeline_config
    )

    return df_train, df_test, df_folds, df_sub


def prepare_loaders(
    df_folds: pd.DataFrame,
    fold: int,
    pipeline_config: global_params.PipelineConfig,
) -> Union[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Prepare the train and validation loaders.

    Args:
        df_folds (pd.DataFrame): The folds dataframe with an additional column "fold".
        fold (int): The fold number.
        pipeline_config (global_params.PipelineConfig): The pipeline config.

    Returns:
        train_loader (torch.utils.data.DataLoader): The train loader.
        valid_loader (torch.utils.data.DataLoader): The validation loader.
        oof (pd.DataFrame): The out of fold dataframe which is the same as the validation dataframe before any changes were made.
    """
    if pipeline_config.global_train_params.debug:
        df_train = df_folds[df_folds["fold"] != fold].sample(
            pipeline_config.loader_params.train_loader["batch_size"]
            * pipeline_config.global_train_params.debug_multiplier,
            random_state=pipeline_config.folds.seed,
        )
        df_valid = df_folds[df_folds["fold"] == fold].sample(
            pipeline_config.loader_params.train_loader["batch_size"]
            * pipeline_config.global_train_params.debug_multiplier,
            random_state=pipeline_config.folds.seed,
        )
        df_oof = df_valid.copy()

    else:
        df_train = df_folds[df_folds["fold"] != fold].reset_index(drop=True)
        df_valid = df_folds[df_folds["fold"] == fold].reset_index(drop=True)
        # Initiate OOF dataframe for this fold (same as df_valid).
        df_oof = df_valid.copy()

    dataset_train = dataset.CustomDataset(
        df_train,
        pipeline_config=pipeline_config,
        transforms=transformation.get_train_transforms(pipeline_config),
        mode="train",
    )
    dataset_valid = dataset.CustomDataset(
        df_valid,
        pipeline_config=pipeline_config,
        transforms=transformation.get_valid_transforms(pipeline_config),
        mode="train",
    )

    # Seeding workers for reproducibility.
    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        **pipeline_config.loader_params.train_loader,
        worker_init_fn=utils.seed_worker,
        generator=g,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset_valid,
        **pipeline_config.loader_params.valid_loader,
        worker_init_fn=utils.seed_worker,
        generator=g,
    )

    # TODO: consider decoupling the oof and loaders, and consider add test loader here for consistency.
    return train_loader, valid_loader, df_oof
