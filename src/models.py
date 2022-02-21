import functools
from pathlib import Path
from typing import Callable, Dict, OrderedDict, Tuple, Union

import timm
import torch
import torchsummary
from config import config, global_params

from src import utils

MODEL_PARAMS = global_params.ModelParams()
LOGS_PARAMS = global_params.LogsParams()
device = config.DEVICE

models_logger = config.init_logger(
    log_file=Path.joinpath(LOGS_PARAMS.LOGS_DIR_RUN_ID, "models.log"),
    module_name="models",
)

# TODO: To check with Ian the best way to put pipeline_config in. The issue is if I put pipeline_config in the constructor
# then it is not easily modifiable outside. It happens that many places I need to define the model as pretrained=False and
# having the pipeline object makes it hard to modify.


class CustomNeuralNet(torch.nn.Module):
    def __init__(
        self,
        model_name: str = MODEL_PARAMS.model_name,
        out_features: int = MODEL_PARAMS.output_dimension,
        in_channels: int = MODEL_PARAMS.input_channels,
        pretrained: bool = MODEL_PARAMS.pretrained,
    ):
        """Construct a new model.

        Args:
            model_name ([type], str): The name of the model to use. Defaults to MODEL_PARAMS.model_name.
            out_features ([type], int): The number of output features, this is usually the number of classes, but if you use sigmoid, then the output is 1. Defaults to MODEL_PARAMS.output_dimension.
            in_channels ([type], int): The number of input channels; RGB = 3, Grayscale = 1. Defaults to MODEL_PARAMS.input_channels.
            pretrained ([type], bool): If True, use pretrained model. Defaults to MODEL_PARAMS.pretrained.
        """
        super().__init__()

        self.in_channels = in_channels
        self.pretrained = pretrained

        self.backbone = timm.create_model(
            model_name, pretrained=self.pretrained, in_chans=self.in_channels
        )
        models_logger.info(
            f"\nModel: {model_name}\nPretrained: {pretrained}\nIn Channels: {in_channels}\n"
        )

        # removes head from backbone: # TODO: Global pool = "avg" vs "" behaves differently in shape, caution!
        self.backbone.reset_classifier(num_classes=0, global_pool="avg")

        # get the last layer's number of features in backbone (feature map)
        self.in_features = self.backbone.num_features
        self.out_features = out_features

        # Custom Head
        # self.single_head_fc = torch.nn.Sequential(
        #     torch.nn.Linear(self.in_features, self.in_features),
        #     torch.nn.ReLU(),
        #     torch.nn.Dropout(p=0.2),
        #     torch.nn.Linear(self.in_features, self.out_features),
        # )
        self.single_head_fc = torch.nn.Sequential(
            torch.nn.Linear(self.in_features, self.out_features),
        )
        self.architecture: Dict[str, Callable] = {
            "backbone": self.backbone,
            "bottleneck": None,
            "head": self.single_head_fc,
        }

    def extract_features(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """Extract the features mapping logits from the model.
        This is the output from the backbone of a CNN.

        Args:
            image (torch.FloatTensor): The input image.

        Returns:
            feature_logits (torch.FloatTensor): The features logits.
        """
        # TODO: To rename feature_logits to image embeddings, also find out what is image embedding.
        feature_logits = self.architecture["backbone"](image)
        return feature_logits

    def forward(self, image: torch.FloatTensor) -> torch.FloatTensor:
        """The forward call of the model.

        Args:
            image (torch.FloatTensor): The input image.

        Returns:
            classifier_logits (torch.FloatTensor): The output logits of the classifier head.
        """

        feature_logits = self.extract_features(image)
        classifier_logits = self.architecture["head"](feature_logits)

        return classifier_logits

    def get_last_layer(self):
        # TODO: Implement this properly.
        """Get the last layer information of TIMM Model.

        Returns:
            [type]: [description]
        """
        last_layer_name = None
        for name, _param in self.model.named_modules():
            last_layer_name = name

        last_layer_attributes = last_layer_name.split(".")  # + ['in_features']
        linear_layer = functools.reduce(
            getattr, last_layer_attributes, self.model
        )
        # reduce applies to a list recursively and reduce
        in_features = functools.reduce(
            getattr, last_layer_attributes, self.model
        ).in_features
        return last_layer_attributes, in_features, linear_layer


def torchsummary_wrapper(
    model: CustomNeuralNet, image_size: Tuple[int, int, int]
) -> torchsummary.model_statistics.ModelStatistics:
    """A torch wrapper to print out layers of a Model.

    Args:
        model (CustomNeuralNet): Model.
        image_size (Tuple[int, int, int]): Image size as a tuple of (channels, height, width).

    Returns:
        model_summary (torchsummary.model_statistics.ModelStatistics): Model summary.
    """

    model_summary = torchsummary.summary(model, image_size)
    return model_summary


def forward_pass(
    loader: torch.utils.data.DataLoader,
    model: CustomNeuralNet,
) -> Union[
    torch.FloatTensor,
    torch.LongTensor,
    torchsummary.model_statistics.ModelStatistics,
]:
    """Performs a forward pass of a tensor through the model.

    Args:
        loader (torch.utils.data.DataLoader): The dataloader.
        model (CustomNeuralNet): Model to be used for the forward pass.

    Returns:
        X (torch.FloatTensor): The input tensor.
        y (torch.LongTensor): The output tensor.
    """
    utils.seed_all()
    model.to(device)

    batch_size, channel, height, width = iter(loader).next()["X"].shape
    image_size = (channel, height, width)

    try:
        models_logger.info("Model Summary:")
        torchsummary.summary(model, image_size)
    except RuntimeError:
        models_logger.debug(f"The channel is {channel}. Check!")

    X = torch.randn((batch_size, *image_size)).to(device)
    y = model(image=X)
    models_logger.info("Forward Pass Successful!")
    models_logger.info(f"X: {X.shape} \ny: {y.shape}")
    models_logger.info(f"X[0][0][0]: {X[0][0][0][0]} \ny[0][0][0]: {y[0][0]}")

    utils.free_gpu_memory(model, X, y)
    return X, y


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cl1 = torch.nn.Linear(25, 60)
        self.cl2 = torch.nn.Linear(60, 16)
        self.fc1 = torch.nn.Linear(16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = torch.nn.ReLU()(self.cl1(x))
        x = torch.nn.ReLU()(self.cl2(x))
        x = torch.nn.ReLU()(self.fc1(x))
        x = torch.nn.ReLU()(self.fc2(x))
        x = torch.nn.LogSoftmax(dim=1)(self.fc3(x))
        return x


class ToySequentialModel(torch.nn.Module):
    # Create a sequential model pytorch same as ToyModel.
    def __init__(self) -> None:
        super().__init__()

        self.backbone = torch.nn.Sequential(
            OrderedDict(
                [
                    ("cl1", torch.nn.Linear(25, 60)),
                    ("cl_relu1", torch.nn.ReLU()),
                    ("cl2", torch.nn.Linear(60, 16)),
                    ("cl_relu2", torch.nn.ReLU()),
                ]
            )
        )

        self.head = torch.nn.Sequential(
            OrderedDict(
                [
                    ("fc1", torch.nn.Linear(16, 120)),
                    ("fc_relu_1", torch.nn.ReLU()),
                    ("fc2", torch.nn.Linear(120, 84)),
                    ("fc_relu_2", torch.nn.ReLU()),
                    ("fc3", torch.nn.Linear(84, 10)),
                    ("fc_log_softmax", torch.nn.LogSoftmax(dim=1)),
                ]
            )
        )

    def forward(self, x):
        """Forward pass of the model.

        Args:
            x ([type]): [description]

        Returns:
            [type]: [description]
        """
        x = self.backbone(x)
        x = self.head(x)
        return x


activation = {}
utils.seed_all()


def get_intermediate_features(name: str) -> Callable:
    """Get the intermediate features of a model. Forward Hook.

    This is using forward hook with reference https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/5

    Args:
        name (str): name of the layer.

    Returns:
        Callable: [description]
    """

    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# The below is testing the forward hook functionalities, especially getting intermediate features.
# Note that both models are same organically but created differently.
# Due to seeding issues, you can check whether they are the same output or not by running them separately.
# We also used assertion to check that the output from model(x) is same as torch.nn.LogSoftmax(dim=1)(fc3_output)

use_sequential_model = False
x = torch.randn(1, 25)
if not use_sequential_model:

    model = ToyModel()

    model.fc2.register_forward_hook(get_intermediate_features("fc2"))
    model.fc3.register_forward_hook(get_intermediate_features("fc3"))
    output = model(x)
    print(activation)
    fc2_output = activation["fc2"]
    print(fc2_output[0])
    fc3_output = activation["fc3"]
    # assert output and logsoftmax fc3_output are the same
    assert torch.allclose(output, torch.nn.LogSoftmax(dim=1)(fc3_output))
else:
    sequential_model = ToySequentialModel()

    # Do this if you want all, if not you can see below.
    # for name, layer in sequential_model.named_modules():
    #     layer.register_forward_hook(get_intermediate_features(name))
    sequential_model.head.fc2.register_forward_hook(
        get_intermediate_features("head.fc2")
    )
    sequential_model.head.fc3.register_forward_hook(
        get_intermediate_features("head.fc3")
    )
    sequential_model_output = sequential_model(x)
    print(activation)
    fc2_output = activation["head.fc2"]
    fc3_output = activation["head.fc3"]
    assert torch.allclose(
        sequential_model_output, torch.nn.LogSoftmax(dim=1)(fc3_output)
    )
