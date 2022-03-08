##### PREPARATIONS
from __future__ import generators, print_function

# libraries
import gc
import pickle
import os
import sys
import urllib.request
import requests
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import cv2
import torch
from scipy.stats import percentileofscore


import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
import wandb
from config import config, global_params
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


from torch._C import device

from src import (
    dataset,
    inference,
    lr_finder,
    metrics,
    models,
    plot,
    prepare,
    trainer,
    transformation,
    utils,
)

# download with progress bar
mybar = None


def show_progress(block_num, block_size, total_size):
    global mybar
    if mybar is None:
        mybar = st.progress(0.0)
    downloaded = block_num * block_size / total_size
    if downloaded <= 1.0:
        mybar.progress(downloaded)
    else:
        mybar.progress(1.0)


import collections
from pathlib import Path
from typing import Any, Dict, List, Union

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import config, global_params
from tqdm.auto import tqdm

from src import dataset, models, utils, trainer


device = config.DEVICE

# TODO: The MODEL_ARTIFACTS_PATH will not be persistent if one is to inference on a new run, so how?
MODEL_ARTIFACTS_PATH = global_params.FilePaths().get_model_artifacts_path()
# 1. Push all inferenced models and oof and submissions to the same folder with the model weights.


def inference_all_folds(
    model: models.CustomNeuralNet,
    state_dicts: List[collections.OrderedDict],
    test_loader: torch.utils.data.DataLoader,
    pipeline_config: global_params.PipelineConfig,
) -> np.ndarray:
    """Inference the model on all K folds.

    Args:
        model (models.CustomNeuralNet): The model to be used for inference. Note that pretrained should be set to False.
        state_dicts (List[collections.OrderedDict]): The state dicts of the models. Generally, K Fold means K state dicts.
        test_loader (torch.utils.data.DataLoader): The dataloader for the test set.

    Returns:
        mean_preds (np.ndarray): The mean of the predictions of all folds.
    """

    model.to(device)
    model.eval()

    with torch.no_grad():
        all_folds_probs = []

        for _fold_num, state in enumerate(state_dicts):
            if "model_state_dict" not in state:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state["model_state_dict"])

            current_fold_probs = []

            for data in tqdm(test_loader, position=0, leave=True):
                images = data["X"].to(device, non_blocking=True)
                test_logits = model(images)
                test_probs = (
                    trainer.get_sigmoid_softmax(pipeline_config)(test_logits)
                    .cpu()
                    .numpy()
                )

                current_fold_probs.append(test_probs)

            current_fold_probs = np.concatenate(current_fold_probs, axis=0)
            all_folds_probs.append(current_fold_probs)
        mean_preds = np.mean(all_folds_probs, axis=0)
    return mean_preds


def inference_streamlit(
    df_test: pd.DataFrame,
    model_dir: Union[str, Path],
    model: Union[models.CustomNeuralNet, Any],
    transform_dict: Dict[str, albumentations.Compose],
    pipeline_config: global_params.PipelineConfig,
    df_sub: pd.DataFrame = None,
    path_to_save: Union[str, Path] = None,
) -> Dict[str, np.ndarray]:

    """Inference the model and perform TTA, if any.

    Dataset and Dataloader are constructed within this function because of TTA.
    model and transform_dict are passed as arguments to enable inferencing multiple different models.

    Args:
        df_test (pd.DataFrame): The test dataframe.
        model_dir (str, Path): model directory for the model.
        model (Union[models.CustomNeuralNet, Any]): The model to be used for inference. Note that pretrained should be set to False.
        transform_dict (Dict[str, albumentations.Compose]): The dictionary of transforms to be used for inference. Should call from get_inference_transforms().
        df_sub (pd.DataFrame, optional): The submission dataframe. Defaults to None.

    Returns:
        all_preds (Dict[str, np.ndarray]): {"normal": normal_preds, "tta": tta_preds}
    """

    if df_sub is None:
        config.logger.info(
            "No submission dataframe detected, setting df_sub to be df_test."
        )
        df_sub = df_test.copy()

    # a dict to keep track of all predictions [no_tta, tta1, tta2, tta3]
    all_preds = {}
    model = model.to(device)

    # Take note I always save my torch models as .pt files. Note we must return paths as str as torch.load does not support pathlib.
    weights = utils.return_list_of_files(
        directory=model_dir, return_string=True, extension=".pt"
    )

    state_dicts = [torch.load(path)["model_state_dict"] for path in weights]

    # Loop over each TTA transforms, if TTA is none, then loop once over normal inference_augs.
    for aug_name, aug_param in transform_dict.items():
        if aug_name != "transforms_test":
            continue  # do not want TTA!
        test_dataset = dataset.CustomDataset(
            df=df_test,
            pipeline_config=pipeline_config,
            transforms=aug_param,
            mode="test",
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **pipeline_config.loader_params.test_loader
        )
        predictions = inference_all_folds(
            model=model,
            state_dicts=state_dicts,
            test_loader=test_loader,
            pipeline_config=pipeline_config,
        )
        print(predictions)
        all_preds[aug_name] = predictions

        ################# To change when necessary depending on the metrics needed for submission #################
        # TODO: Consider returning a list of predictions ranging from np.argmax to preds, probs etc, and this way we can use whichever from the output? See my petfinder for more.
        df_sub[pipeline_config.folds.class_col_name] = predictions[:, 1]

    # for each value in the dictionary all_preds, we need to take the mean of all the values and assign it to a df and save it.
    df_sub[pipeline_config.folds.class_col_name] = np.mean(
        list(all_preds.values()), axis=0
    )[:, 1]

    return all_preds


##### CONFIG

# page config
st.set_page_config(
    page_title="Score your pet!",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)


##### HEADER

# title
st.title("How cute is your pet?")

# image cover
cover_image = Image.open(
    requests.get(
        "https://storage.googleapis.com/kaggle-competitions/kaggle/25383/logos/header.png?t=2021-08-31-18-49-29",
        stream=True,
    ).raw
)
st.image(cover_image)

# description
st.write(
    "This app uses deep learning to estimate a pawpularity score of custom pet photos. Pawpularity is a metric used by [PetFinder](https://petfinder.my/) to judge the pet's attractiveness, which translates to more clicks for the pet profile."
)


##### PARAMETERS

# header
st.header("Score your own pet")

# photo upload
pet_image = st.file_uploader("1. Upload your pet photo.")
if pet_image is not None:
    print(pet_image.name)
    # check image format
    image_path = "app/tmp/" + pet_image.name
    if (
        (".jpg" not in image_path)
        and (".JPG" not in image_path)
        and (".jpeg" not in image_path)
        and (".bmp" not in image_path)
    ):
        st.error("Please upload .jpeg, .jpg or .bmp file.")
    else:

        # save image to folder
        with open(image_path, "wb") as f:
            f.write(pet_image.getbuffer())

        # display pet image
        st.success("Pet photo uploaded.")

# privacy toogle
choice = st.radio(
    "2. Make the result public?",
    [
        "Yes. Others may see your pet photo.",
        "No. Scoring will be done privately.",
    ],
)

# model selection
model_name = st.selectbox(
    "3. Choose a model for scoring your pet.",
    ["EfficientNet B3", "Swin Transformer"],
)


##### MODELING

# compute pawpularity
if st.button("Compute pawpularity"):

    # check if image is uploaded
    if pet_image is None:
        st.error("Please upload a pet image first.")

    else:

        # # specify paths
        # if model_name == "EfficientNet B3":
        #     weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/enet_b3.pth"
        #     model_path = "app/models/enet_b3/"
        # elif model_name == "EfficientNet B5":
        #     weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/enet_b5.pth"
        #     model_path = "app/models/enet_b5/"
        # elif model_name == "Swin Transformer":
        #     weight_path = "https://github.com/kozodoi/pet_pawpularity/releases/download/0.1/swin_base.pth"
        #     model_path = "app/models/swin_base/"

        # # download model weights
        # if not os.path.isfile(model_path + "pytorch_model.pth"):
        #     with st.spinner(
        #         "Downloading model weights. This is done once and can take a minute..."
        #     ):
        #         urllib.request.urlretrieve(
        #             weight_path, model_path + "pytorch_model.pth", show_progress
        #         )
        # TODO: model_dir is defined hardcoded, consider be able to pull the exact path from the saved logs/models from wandb even?

        # Define global parameters to pass in PipelineConfig
        FILES = global_params.FilePaths()
        LOADER_PARAMS = global_params.DataLoaderParams()
        FOLDS = global_params.MakeFolds()
        TRANSFORMS = global_params.AugmentationParams()
        MODEL_PARAMS = global_params.ModelParams()

        GLOBAL_TRAIN_PARAMS = global_params.GlobalTrainParams()
        WANDB_PARAMS = global_params.WandbParams()
        LOGS_PARAMS = global_params.LogsParams()

        CRITERION_PARAMS = global_params.CriterionParams()
        SCHEDULER_PARAMS = global_params.SchedulerParams()
        OPTIMIZER_PARAMS = global_params.OptimizerParams()

        utils.seed_all(FOLDS.seed)

        INFERENCE_TRANSFORMS = global_params.AugmentationParams(image_size=256)

        # INFERENCE_MODEL_PARAMS = global_params.ModelParams()
        inference_pipeline_config = global_params.PipelineConfig(
            files=FILES,
            loader_params=LOADER_PARAMS,
            folds=FOLDS,
            transforms=INFERENCE_TRANSFORMS,
            model_params=MODEL_PARAMS,
            global_train_params=GLOBAL_TRAIN_PARAMS,
            wandb_params=WANDB_PARAMS,
            logs_params=LOGS_PARAMS,
            criterion_params=CRITERION_PARAMS,
            scheduler_params=SCHEDULER_PARAMS,
            optimizer_params=OPTIMIZER_PARAMS,
        )

        # @Step 1: Download and load data.
        df_train, df_test, df_folds, df_sub = prepare.prepare_data(
            inference_pipeline_config
        )

        model_dir = Path(
            r"C:\Users\reighns\reighns_ml\kaggle\siim_isic_melanoma_classification\app\model_weights\tf_efficientnet_b1_ns_tf_efficientnet_b1_ns_5_folds_9qhxwbbq"
        )

        weights = utils.return_list_of_files(
            directory=model_dir, return_string=True, extension=".pt"
        )

        model = models.CustomNeuralNet(
            model_name="tf_efficientnet_b1_ns",
            out_features=2,
            in_channels=3,
            pretrained=False,
        ).to(device)

        transform_dict = transformation.get_inference_transforms(
            pipeline_config=inference_pipeline_config,
        )

        # compute predictions
        with st.spinner("Computing prediction..."):

            # clear memory
            gc.collect()

            predictions = inference.inference(
                df_test=df_test,
                model_dir=model_dir,
                model=model,
                df_sub=df_test,
                transform_dict=transform_dict,
                pipeline_config=inference_pipeline_config,
                path_to_save=model_dir,
            )

            # process pet image
            pet_image = cv2.imread(image_path)
            pet_image = cv2.cvtColor(pet_image, cv2.COLOR_BGR2RGB)
            image = augs(image=pet_image)["image"]

            # compute prediction
            pred = model(torch.unsqueeze(image, 0))
            score = np.round(100 * pred.detach().numpy()[0][0], 2)

            # compute percentile
            oof = pd.read_csv(model_path + "oof.csv")
            percent = np.round(
                percentileofscore(oof["pred"].values * 100, score), 2
            )

            # display results
            col1, col2 = st.columns(2)
            col1.image(cv2.resize(pet_image, (256, 256)))
            col2.metric("Pawpularity", score)
            col2.metric("Percentile", str(percent) + "%")
            col2.write(
                "**Note:** pawpularity ranges from 0 to 100. Scroll down to read more about the metric and the implemented models."
            )

            # clear memory
            del config, model, augs, image
            gc.collect()

            # celebrate
            st.success("Well done! Thanks for scoring your pet :)")
