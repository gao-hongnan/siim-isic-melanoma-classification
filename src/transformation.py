from typing import Dict, Union

import numpy as np
import torch
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from config import global_params


def get_train_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on training data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for training set.
    """

    return albumentations.Compose(
        [
            albumentations.RandomResizedCrop(
                height=pipeline_config.transforms.image_size,
                width=pipeline_config.transforms.image_size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333333333333333),
                p=1.0,
            ),
            albumentations.VerticalFlip(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightness(limit=0.2, p=0.75),
            albumentations.RandomContrast(limit=0.2, p=0.75),
            albumentations.OneOf(
                [
                    albumentations.MotionBlur(blur_limit=5),
                    albumentations.MedianBlur(blur_limit=5),
                    albumentations.GaussianBlur(blur_limit=5),
                    albumentations.GaussNoise(var_limit=(5.0, 30.0)),
                ],
                p=0.7,
            ),
            albumentations.OneOf(
                [
                    albumentations.OpticalDistortion(distort_limit=1.0),
                    albumentations.GridDistortion(
                        num_steps=5, distort_limit=1.0
                    ),
                    albumentations.ElasticTransform(alpha=3),
                ],
                p=0.7,
            ),
            albumentations.CLAHE(clip_limit=4.0, p=0.7),
            albumentations.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.5,
            ),
            albumentations.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=0,
                p=0.85,
            ),
            albumentations.Cutout(
                max_h_size=int(pipeline_config.transforms.image_size * 0.375),
                max_w_size=int(pipeline_config.transforms.image_size * 0.375),
                num_holes=1,
                p=0.5,
            ),
            albumentations.Normalize(
                mean=pipeline_config.transforms.mean,
                std=pipeline_config.transforms.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_valid_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on validation data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for validation set.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(
                pipeline_config.transforms.image_size,
                pipeline_config.transforms.image_size,
            ),
            albumentations.Normalize(
                mean=pipeline_config.transforms.mean,
                std=pipeline_config.transforms.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_gradcam_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> albumentations.core.composition.Compose:
    """Performs Augmentation on gradcam data.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        albumentations.core.composition.Compose: The transforms for gradcam.
    """
    return albumentations.Compose(
        [
            albumentations.Resize(
                pipeline_config.transforms.image_size,
                pipeline_config.transforms.image_size,
            ),
            albumentations.Normalize(
                mean=pipeline_config.transforms.mean,
                std=pipeline_config.transforms.std,
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ]
    )


def get_inference_transforms(
    pipeline_config: global_params.PipelineConfig,
) -> Dict[str, albumentations.core.composition.Compose]:
    """Performs Augmentation on test dataset.

    Remember tta transforms need resize and normalize.

    Args:
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        image_size (int, optional): The image size. Defaults to TRANSFORMS.image_size.
        mean (List[float], optional): The mean. Defaults to TRANSFORMS.mean.
        std (List[float], optional): The std. Defaults to TRANSFORMS.std.

    Returns:
        transforms_dict (Dict[str, albumentations.core.composition.Compose]): Returns the transforms for inference in a dictionary which can hold TTA transforms.
    """

    transforms_dict = {
        "transforms_test": albumentations.Compose(
            [
                albumentations.Resize(
                    pipeline_config.transforms.image_size,
                    pipeline_config.transforms.image_size,
                ),
                albumentations.Normalize(
                    mean=pipeline_config.transforms.mean,
                    std=pipeline_config.transforms.std,
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        ),
        # "tta_hflip": albumentations.Compose(
        #     [
        #         albumentations.HorizontalFlip(p=1.0),
        # albumentations.Resize(
        #     pipeline_config.transforms.image_size,
        #     pipeline_config.transforms.image_size,
        # ),
        # albumentations.Normalize(
        #     mean=pipeline_config.transforms.mean,
        #     std=pipeline_config.transforms.std,
        #     max_pixel_value=255.0,
        #     p=1.0,
        # ),
        #         ToTensorV2(),
        #     ]
        # ),
    }

    return transforms_dict


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    pipeline_config: global_params.PipelineConfig,
) -> torch.Tensor:
    """Implements mixup data augmentation.

    Args:
        x (torch.Tensor): The input tensor.
        y (torch.Tensor): The target tensor.
        pipeline_config (global_params.PipelineConfig): The pipeline config.
        mixup_params (TRANSFORMS, optional): [description]. Defaults to TRANSFORMS.mixup_params.

    Returns:
        torch.Tensor: [description]
    """

    mixup_params = pipeline_config.transforms.mixup_params

    # TODO: https://www.kaggle.com/reighns/petfinder-image-tabular check this to add z if there are dense targets.
    assert (
        mixup_params["mixup_alpha"] > 0
    ), "Mixup alpha must be greater than 0."
    assert (
        x.size(0) > 1
    ), "Mixup requires more than one sample as at least two samples are needed to mix."

    if mixup_params["mixup_alpha"] > 0:
        lambda_ = np.random.beta(
            mixup_params["mixup_alpha"], mixup_params["mixup_alpha"]
        )
    else:
        lambda_ = 1

    batch_size = x.size()[0]
    if mixup_params["use_cuda"] and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lambda_ * x + (1 - lambda_) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lambda_


def mixup_criterion(
    criterion: Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss],
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lambda_: float,
) -> torch.Tensor:
    """Implements mixup criterion.

    Args:
        criterion (Union[torch.nn.BCEWithLogitsLoss, torch.nn.CrossEntropyLoss]): The loss function.
        logits (torch.Tensor): [description]
        y_a (torch.Tensor): [description]
        y_b (torch.Tensor): [description]
        lambda_ (float): [description]

    Returns:
        torch.Tensor: [description]
    """
    return lambda_ * criterion(logits, y_a) + (1 - lambda_) * criterion(
        logits, y_b
    )
