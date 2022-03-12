<div align="center">
<h1>SIIM-ISIC Melanoma Classification, 2020</a></h1>
by Hongnan Gao
<br>
</div>

- [SIIM-ISIC Melanoma Classification](#siim-isic-melanoma-classification)
- [Establish Metrics](#establish-metrics)
  - [Benefit Structure](#benefit-structure)
  - [ROC](#roc)
  - [Brier Score Loss](#brier-score-loss)
  - [What can we explore?](#what-can-we-explore)
- [Validation and Resampling Strategy](#validation-and-resampling-strategy)
  - [How should we split out data into folds?](#how-should-we-split-out-data-into-folds)
  - [Cross-Validation Workflow](#cross-validation-workflow)
- [Transfer Learning](#transfer-learning)
  - [Fine-Tuning](#fine-tuning)
  - [Feature Extraction](#feature-extraction)
- [Preprocessing](#preprocessing)
  - [Mean and Standard Deviation](#mean-and-standard-deviation)
  - [Channel Distribution](#channel-distribution)
  - [Let the Model tell you where went wrong!](#let-the-model-tell-you-where-went-wrong)
- [Augmentations](#augmentations)
  - [Train-Time Augmentation](#train-time-augmentation)
  - [Test-Time Augmentation](#test-time-augmentation)
- [Optimizer, Scheduler and Loss](#optimizer-scheduler-and-loss)
  - [Optimizer](#optimizer)
  - [Scheduler](#scheduler)
  - [Loss](#loss)
- [Model Architectures, Training Parameters](#model-architectures-training-parameters)
  - [No Meta Data Model Architecture](#no-meta-data-model-architecture)
  - [Meta Data Model Architecture](#meta-data-model-architecture)
  - [Activation Functions](#activation-functions)
- [Ensemble Theory](#ensemble-theory)
  - [Mean Blending](#mean-blending)
  - [Forward Ensembling](#forward-ensembling)
- [Error Analysis using Grad-CAM](#error-analysis-using-grad-cam)
- [Next Steps](#next-steps)
- [References](#references)


## SIIM-ISIC Melanoma Classification

This competition is hosted on Kaggle and the [description and overview is stated below](https://www.kaggle.com/c/siim-isic-melanoma-classification/overview).

Skin cancer is the most prevalent type of cancer. Melanoma, specifically, is responsible for 75% of skin cancer deaths, despite being the least common skin cancer. The American Cancer Society estimates over 100,000 new melanoma cases will be diagnosed in 2020. It's also expected that almost 7,000 people will die from the disease. As with other cancers, early and accurate detection—potentially aided by data science—can make treatment more effective.

Currently, dermatologists evaluate every one of a patient's moles to identify outlier lesions or “ugly ducklings” that are most likely to be melanoma. Existing AI approaches have not adequately considered this clinical frame of reference. Dermatologists could enhance their diagnostic accuracy if detection algorithms take into account “contextual” images within the same patient to determine which images represent a melanoma. If successful, classifiers would be more accurate and could better support dermatological clinic work.

As the leading healthcare organization for informatics in medical imaging, the Society for Imaging Informatics in Medicine (SIIM)'s mission is to advance medical imaging informatics through education, research, and innovation in a multi-disciplinary community. SIIM is joined by the International Skin Imaging Collaboration (ISIC), an international effort to improve melanoma diagnosis. The ISIC Archive contains the largest publicly available collection of quality-controlled dermoscopic images of skin lesions.

In this competition, you’ll identify melanoma in images of skin lesions. In particular, you’ll use images within the same patient and determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.

Melanoma is a deadly disease, but if caught early, most melanomas can be cured with minor surgery. Image analysis tools that automate the diagnosis of melanoma will improve dermatologists' diagnostic accuracy. Better detection of melanoma has the opportunity to positively impact millions of people.

## Establish Metrics

After understanding the problem better, we should probably define a metric to optimize. As usual, this step should be closely tied to business problem. We were already given a metric score by the competition host and let us understand it better.

Recall that we wish to have a well-calibrated model, the intuition is that a high performance model may not output meaningful probabilities, even if they can have extremely good performance score.

Consider a model that outputs logits of $0.51$ when y_true is 1 and $0.49$ otherwise, then a decision threshold of $0.5$ guarantees an accuracy of $100\%$, we have no complaints here if we have no issue with our threshold if our only goal is to have a high scoring model. However, if in medical case, where doctor wants to understand "probablistically" the survival of a patient, then we might want to turn into logits probs. But apparently the example here holds almost no meaning, when compared to a "well calibrated model", more concretely.

```python
y_true = [0, 0, 1, 1]
y_prob_uncalibrated = [0.49, 0.49, 0.51, 0.51]
y_prob_calibrated = [0.1, 0.45, 0.99, 0.6]
```

both models give $100\%$ accuracy, but the latter (assuming calibrated), can give us a laymen idea that ok this patient has 0.99 chance and the other patient 0.6 chance of surviving etc.

### Benefit Structure

One can introduce a **benefit structure** with relevant cost-benefit assignment.

- TP: + 100
- FN: -1000
- FP: -10
- TP+FP: -1 (screening for example)

With each TP, we net a profit of 100, and with each FN, we lose -1000, FP loses -10 and whenever the patient get predicted to die (1), send for further screening -1. So towards the end, we can have:

$$
cost = 100*TP - 1000 * FN - 10 * FP - 1 * (TP+FP)
$$

This structure helps us decide which metrics to choose.

### ROC

Definition: The basic (non-probablistic intepretation) of ROC is graph that plots the True Positive Rate on the y-axis and False Positive Rate on the x-axis parametrized by a threshold vector `t`. We then look at the area under the ROC curve (AUROC) to get an overall performance measure. Note that TPR is `recall`.

- TPR (recall) = TP / (TP + FN)
- FPR = FP / (FP + TN)
- Threshold invariant
    - The ROC looks at the performance of a model hypothesis at all thresholds. This is better than just optimizing **recall** which only looks at a fixed threshold.
- Scale Invariant
    - Not necessarily a good thing in this context, as this makes ROC a semi-proper scoring metric, that is, it takes in non-calibrated scores and perform well. **The below code shows that as long as the order is preserved, `y2` and `y4` make zero difference in the outcome. In this case, the doctor may not be able to have a “confidence” level of how likely the patient is going to survive.**
        
        ```python
        y1 = [1,0,1,0]
        y2 = [0.52,0.51,0.52,0.51]
        y3 = [52,51,52,51]
        y4 = [0.99, 0.51, 0.98, 0.51]
        uncalibrated_roc = roc(y1,y2) == roc(y1,y3) == roc(y1, y4)
        print(f"{uncalibrated_roc}") -> 1.0
        ```
        
    - This brings us to the next point.

More info in notebook.

### Brier Score Loss

Brier Score computes the squared difference between the probability of a prediction and its actual outcome. Intuitively, this score punishes “unconfident and neutral” probability logits. If a model consistently spits out probability that is near 0.5, then this score will be large. 

- Proper scoring
    - Tells us if the scores output are well calibrated.
    - If not well calibrated, prompt us to either use a different model that calibrated well, or to perform calibration on the model itself.
    - Logistic regression produces natural well calibrated probabilities since it optimizes the log-loss (ce loss), in fact, I think MLE models should always produce well calibrated probabilities since behind the scene it is minimizing KL divergence between ground truth distribution P and estimated distribution Q.
    - It follows that models like DT do not produce well calibrated probabilities.

More info in notebook.

### What can we explore?

- Did not provide insight if Precision-recall curve and if it may be well posed for this problem than ROC since there is some class imbalance.
- Did not go into details on calibration methods, in fact, models like RF are not well calibrated by construction. [https://scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html)

## Validation and Resampling Strategy

### How should we split out data into folds?

We should examine the data for a few factors:

1. Is the data $\mathcal{X}$ imbalanced?
2. Is the data $\mathcal{X}$ generated in a **i.i.d.** manner, more specifically, if I split $\mathcal{X}$ to $\mathcal{X}_{train}$ and $\mathcal{X}_{val}$, can we ensure that $\mathcal{X}_{val}$ has no dependency on $\mathcal{X}_{train}$?

We came to the conclusion:

1. Yes, the data is severely imbalanced in which there are only around $2\%$ of positive (malignant) samples. Therefore, a stratified cross validation is reasonable. `StratifiedKFold` ensures that relative class frequencies is approximately preserved in each train and validation fold. More concretely, we will not experience the scenario where $X_{train}$ has $m^{+}$ and $m^{-}$ positive and negative samples, but $X_{val}$ has only $p^{+}$ positive samples only and 0 negative samples, simply due to the scarcity of negative samples.

2. In medical imaging, it is a well known fact that most of the data contains patient level repeatedly. To put it bluntly, if I have 100 samples, and according to **PatientID**, we see that the id 123456 (John Doe) appeared 20 times, this is normal as a patient can undergo multiple settings of say, X-rays. If we allow John Doe's data to appear in both train and validation set, then this poses a problem of information leakage, in which the data is no longer **i.i.d.**. One can think of each patient has an "unique, underlying features" which are highly correlated across their different samples. As a result, it is paramount to ensure that amongst this 3255 unique patients, we need to ensure that each unique patients' images **DO NOT** appear in the validation fold. That is to say, if patient John Doe has 100 X-ray images, but during our 5-fold splits, he has 70 images in Fold 1-4, while 30 images are in Fold 5, then if we were to train on Fold 1-4 and validate on Fold 5, there may be potential leakage and the model will predict with confidence for John Doe's images. This is under the assumption that John Doe's data does not fulfill the i.i.d proces

---

With the above consideration, we will use `StratifiedGroupKFold` where $K = 5$ splits. There wasn't this splitting function in scikit-learn at the time of competition and as a result, we used a custom written (by someone else) `RepeatedStratifiedGroupKFold` function and just set `n_splits = 1` to get **StratifiedGroupKFold** (yes we cannot afford to repeated sample, so setting the split to be 1 will collapse the repeated function to just the normal stratified group kfold). However, as of 2022, this function is readily available in the **Scikit-Learn** library.

To recap, we applied stratified logic such that each train and validation set has an **equal** weightage of positive and negative samples. We also grouped the patients in the process such that patient $i$ will not appear in both training and validation set.

---

> It is worth mentioning the famous Kaggler Chris Deotte went one step further to **Triple Stratify** the data where he balanced patient count distribution. One can read more [here](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526).

---

### Cross-Validation Workflow

To recap, we have the following:

- **Training Set ($X_{\text{train}}$)**: This will be further split into K validation sets during our cross-validation. This set is used to fit a particular hypothesis $h \in \mathcal{H}$.
- **Validation Set ($X_{\text{val}}$)**: This is split from our $X_{\text{train}}$ during cross-validation. This set is used for model selection (i.e. find best hyperparameters, attempt to produce a best hypothesis $g \in \mathcal{H}$).
- **Test Set ($X_{\text{test}}$)**: This is an unseen test set, and we will only use it after we finish tuning our model/hypothesis. Suppose we have a final best model $g$, we will use $g$ to predict on the test set to get an estimate of the generalization error (also called out-of-sample error).

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/supervised_learning/classification/aiap-coronary-artery-disease/data/images/cv.PNG' width="500"/>
<figcaption align = "center"><b>Pipeline.</b></figcaption>
</figure>

---

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/supervised_learning/classification/breast-cancer-wisconsin/data/images/grid_search_workflow.png' width="500"/>
<figcaption align = "center"><b>Courtesy of scikit-learn on a typical Cross-Validation workflow.</b></figcaption>
</figure>

## Transfer Learning

Traditionally, training on **ImageNet** weights is a good choice to start. In the event that our training set has a very different distribution of what's inside **ImageNet**, the model may take a while to converge, even if we finetune it. The intuition is simple, **ImageNet** was trained on many common items in life, and none of them resemble closely to the image structures of **Melanoma Images**. Consequently, the model may have a hard time detecting shapes and details from these medical images.

We can of course unfreeze all the layers and retrain them from scratch, using various backbones, however, due to limited hardware, we decided it is best to first check if **ImageNet** yields good results, if not, we can explore weights that were originally trained on skin cancer images. 

The community used a few models and found out that the **EfficientNet** variants yielded the best results on this set of training images using **ImageNet** and hence we adopt the **EfficientNet** family moving forward. Examining the Grad-CAM of the models revealed that this family of models not only focus on the center nucleus of the skin image but also corners, perhaps they capture something other models don't? We will compare them briefly later.

### Fine-Tuning

Instead of random initialization, we initialize the network with a pretrained network, like the one that is trained on imagenet 1000 dataset. This is what we will be doing. References below.

### Feature Extraction

ConvNet as fixed feature extractor: Here, we will freeze the weights for all of the network except that of the final fully connected layer. This last fully connected layer is replaced with a new one with random weights and only this layer is trained.

## Preprocessing

Most preprocessing techniques we do in an image recognition competition is mostly as follows:

### Mean and Standard Deviation

- Perform **mean and std** for the dataset given to us. Note that this step may make sense on paper, but empirically, using imagenet's default mean std will always work as well, if not better. You can read my [blog post here](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/image_normalization/Image_Normalization_and_Standardization/)"
    - Imagenet on RGB: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]


### Channel Distribution

This is usually done to check for "surprises". More specifically, I remembered once that someone trained a CNN on the blood cells dataset (red, white blood cells etc), as a beginner who just came out from MNIST, he/she grayscaled the images and yielded poor results. This is because one distinct way for the model to differentiate these cells might be because of the colors of the cells.

### Let the Model tell you where went wrong!

Alternatively, the issues are not obvious and we can use tools like Grad-CAM to see where our model is looking to deduce why the model is performing poorly.

## Augmentations

We know that augmentation is central in an image competition, as essentially we are adding more data into the training process, effectively reducing overfitting.

Heavy augmentations are used during Train-Time-Augmentation. But during Test-Time-Augmentation, we used the same set of training augmentations to inference with $100\%$ probability.

### Train-Time Augmentation

Community power. We made use of some innovative augmentations:

- [AdvancedHairAugmentation](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159176) where hairs were randomly added to the image and
- [Microscope](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/159476) where images were made to look as if they were taken from a microscope.

Both of these augmentations provided a steady increase in CV and LB.

```python
albumentations.Compose(
    [
        AdvancedHairAugmentation(
            hairs_folder=pipeline_config.transforms.hairs_folder
        ),
        albumentations.RandomResizedCrop(
            height=pipeline_config.transforms.image_size,
            width=pipeline_config.transforms.image_size,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333333333),
            p=1.0,
        ),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Cutout(
            max_h_size=int(pipeline_config.transforms.image_size * 0.375),
            max_w_size=int(pipeline_config.transforms.image_size * 0.375),
            num_holes=1,
            p=0.3,
        ),
        Microscope(p=0.5),
        albumentations.Normalize(
            mean=pipeline_config.transforms.mean,
            std=pipeline_config.transforms.std,
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ]
)
```

### Test-Time Augmentation

The exact same set of augmentations were used in inference. Not all TTAs provided a increase in score.

## Optimizer, Scheduler and Loss

### Optimizer

We used good old `AdamW` keeping in mind the rule of thumb that if batch size increase by a factor of 2, learning rate should increase by a factor of 2 as well.

```python
optimizer_name: str = "AdamW"
optimizer_params: Dict[str, Any] = field(
    default_factory=lambda: {
        "lr": 1e-4,
        "betas": (0.9, 0.999),
        "amsgrad": False,
        "weight_decay": 1e-6,
        "eps": 1e-08,
    }
)
```

### Scheduler

We used the following settings:

```python
scheduler_name: str = "CosineAnnealingWarmRestarts"  # Debug
if scheduler_name == "CosineAnnealingWarmRestarts":
    scheduler_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "T_0": 10,
            "T_mult": 1,
            "eta_min": 1e-6,
            "last_epoch": -1,
        }
    )
```

One should note that `OneCycleLR` is very popular and yields good results with shorter convergence time.

### Loss

We used `CrossEntropyLoss` loss with default parameters. Read more in my [blog post](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/fundamentals/loss_functions/cross_entropy_loss/cross_entropy_loss_from_scratch/).

```python
train_criterion_name: str = "CrossEntropyLoss"
train_criterion_params: Dict[str, Any] = field(
    default_factory=lambda: {
        "weight": None,
        "size_average": None,
        "ignore_index": -100,
        "reduce": None,
        "reduction": "mean",
        "label_smoothing": 0.0,
    }
)
```





## Model Architectures, Training Parameters

### No Meta Data Model Architecture

For models that did not make use of meta data, we have the following architecture.

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/no_meta_model_architecure.svg' width="800"/>
<figcaption align = "center"><b>No Meta Data Model Architecture.</b></figcaption>
</figure>


### Meta Data Model Architecture

For models that did made use of meta data, we have the following architecture.

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/meta_model_architecture.svg' width="800"/>
<figcaption align = "center"><b>Meta Data Model Architecture.</b></figcaption>
</figure>


We concat the flattened feature maps with the meta features: 

```python
Meta Features: ['sex', 'age_approx', 'site_head/neck', 'site_lower extremity', 'site_oral/genital', 'site_palms/soles', 'site_torso', 'site_upper extremity', 'site_nan']
```

and the meta features has its own sequential layers as ANN:

```python
OrderedDict(
    [
        (
            "fc1",
            torch.nn.Linear(self.num_meta_features, 512),
        ),
        (
            "bn1",
            torch.nn.BatchNorm1d(512),
        ),
        (
            "swish1",
            torch.nn.SiLU(),
        ),
        (
            "dropout1",
            torch.nn.Dropout(p=0.3),
        ),
        (
            "fc2",
            torch.nn.Linear(512, 128),
        ),
        (
            "bn2",
            torch.nn.BatchNorm1d(128),
        ),
        (
            "swish2",
            torch.nn.SiLU(),
        ),
    ]
)
```




For example:

- image shape: $[32, 3, 256, 256]$
- meta_inputs shape: $[32, 9]$ we have 9 features.
- feature_logits shape: $[32, 1280]$ flattened feature maps at the last conv layer.
- meta_logits shape: $[32, 128]$ where we passed in a small sequential ANN for the meta data.
- concat_logits shape: $[32, 1280 + 128]$

```python
if self.use_meta:
    # from cnn images
    feature_logits = self.extract_features(image)

    # from meta features
    meta_logits = self.meta_layer(meta_inputs)

    # concatenate
    concat_logits = torch.cat((feature_logits, meta_logits), dim=1)

    # classifier head
    classifier_logits = self.architecture["head"](concat_logits)
```


### Activation Functions

As we all know, activation functions are used to transform a neurons' linearity to non-linearity and decide whether to "fire" a neuron or not.

When we design or choose an activation function, we need to ensure the follows:

- (Smoothness) Differentiable and Continuous: For example, the sigmoid function is continuous and hence differentiable. If the property is not fulfilled, we might face issues as backpropagation may not be performed properly since we cannot differentiate it.If you notice, the heaviside function is not. We cant perform GD using the HF as we cannot compute gradients but for the logistic function we can. The gradient of sigmoid function g is g(1-g) conveniently

- Monotonic: This helps the model to converge faster. But spoiler alert, Swish is not monotonic.

The properties of Swish are as follows:

- Bounded below: It is claimed in the paper it serves as a strong regularization.
- Smoothness: More smooth than ReLU which allows the model to optimize better, the error landscape, when smoothed, is easier to traverse in order to find a minima. An intuitive idea is the hill again, imagine you traverse down your neighbourhood hill, vs traversing down Mount Himalaya.

```python
# Import matplotlib, numpy and math
import matplotlib.pyplot as plt
import numpy as np
import math

def swish(x):
    sigmoid =  1/(1 + np.exp(-x))
    swish = x * sigmoid
    return swish

epsilon = 1e-20
x = np.linspace(-100,100, 100)
z = swish(x)
print(z)
print(min(z))

plt.plot(x, z)
plt.xlabel("x")
plt.ylabel("Swish(X)")

plt.show()
```

## Ensemble Theory

### Mean Blending

This is just simple mean blending.

### Forward Ensembling

We made use of the [Forward Ensembling](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614) idea from Chris in SIIM-ISIC Melanoma Classification back in August 2020, I modified the code for this specific task. A simple description is as follows, modified from Chris, with more mathematical notations.

1. We start off with a dataset $\mathcal{D} = X \times y$ where it is sampled from the true population $\mathcal{X} \times \mathcal{Y}$.
2. We apply KFold (5 splits) to the dataset, as illustrated in the diagram. 
3. We can now train five different hypothesis $h_{F1}, h_{F2},...,h_{F5}$, where $h_{F1}$ is trained on Fold 2 to Fold 5 and predict on Fold 1, $h_{F2}$ is trained on Fold 1,3,4,5 and predict on Fold 2. The logic follows for all 5 hypothesis.
4. Notice that in the five models, we are predicting on a unique validation fold, and as a result, after we trained all 5 folds, we will have the predictions made on the whole training set (F1-F5). This predictions is called the Out-of-Fold predictions.
5. We then go a step further and calculate the AUC score with the OOF predictions with the ground truth to get the OOF AUC. We save it to a csv or dataframe called **oof_1.csv**, subsequent oof trained on different hypothesis space should be named **oof_i.csv** where $i \in [2,3,...]$.
6. After we trained all 5 folds, we will use $h_{1}$ to predict on $X_{test}$ and obtain predictions $Y_{\text{h1 preds}}$, we then use $h_{2}$ to predict on $X_{test}$ and obtain predictions $Y_{\text{h2 preds}}$, we do this for all five folds and finally $Y_{\text{final preds}} = \dfrac{1}{5}\sum_{i=1}^{5}Y_{\text{hi preds}}$. This is a typical pipeline in most machine learning problems. We save this final predictions as **sub_1.csv**, subsequence predictions trained on different hypothesis space should be named **sub_i.csv** where $i \in [2,3,...]$.
7. Now if we train another model, a completely different hypothesis space is used, to be more pedantic, we denote the previous model to be taken from the hypothesis space $\mathcal{H}_{1}$, and now we move on to $\mathcal{H}_{2}$. We repeat step 1-6 on this new model (Note that you are essentially training 10 "models" now since we are doing KFold twice, and oh, please set the seed of KFold to be the same, it should never be the case that both model comes from different splitting seed for apparent reasons).

---

Here is the key (given the above setup with 2 different models trained on 5 folds):

1. Normally, most people do a simple mean ensemble, that is $\dfrac{Y_{\text{final preds H1}} + Y_{\text{final preds H2}}}{2}$. This works well most of the time as we trust both model holds equal importance in the final predictions.
2. One issue may be that certain models should be weighted more than the rest, we should not simply take Leaderboard feedback score to judge the weight assignment. A general heuristic here is called Forward Selection.
3. (Extracted from Chris) Now say that you build 2 models (that means that you did 5 KFold twice). You now have oof_1.csv, oof_2.csv, sub_1.csv, and sub_2.csv. How do we blend the two models? We find the weight w such that `w * oof_1.predictions + (1-w) * oof_2.predictions` has the largest AUC.

```python
all = []
for w in [0.00, 0.01, 0.02, ..., 0.98, 0.99, 1.00]:
    ensemble_pred = w * oof_1.predictions + (1-w) * oof_2.predictions
    ensemble_auc = roc_auc_score( oof.target , ensemble_pred )
    all.append( ensemble_auc )
best_weight = np.argmax( all ) / 100.
```

Then we can assign the best weight like:

```python
final_ensemble_pred = best_weight * sub_1.target + (1-best_weight) * sub_2.target
```

Read more from my blog post in references below.

## Error Analysis using Grad-CAM

There is some distinct difference when Grad-CAM is applied to different models, which can help us do error analysis.

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/resnet50d_image.PNG' width="800"/>
<figcaption align = "center"><b>Grad-CAM of ResNet50d</b></figcaption>
</figure>

<figure>
<img src='https://storage.googleapis.com/reighns/reighns_ml_projects/docs/projects/SIIM-ISIC%20Melanoma%20Classification/images/tf_efficientnet_b1_ns_image.PNG' width="800"/>
<figcaption align = "center"><b>Grad-CAM of EfficietNet</b></figcaption>
</figure>

For more info on [Grad-CAM](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/neural_network_interpretation/05_gradcam_and_variants/gradcam_explained/), see my blog post.

## Next Steps

- MLOps (Weights & Biases for experiment tracking)
- Model Persistence
- Benefit Structure

## References

- [Image Normalization](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/image_normalization/Image_Normalization_and_Standardization/)
- [Triple Stratified Leak-Free KFold CV](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/165526)
- [Transfer Learning PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Transfer Learning TensorFlow](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Cross-Entropy Loss](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/fundamentals/loss_functions/cross_entropy_loss/cross_entropy_loss_from_scratch/)
- [Forward Ensemble](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/ensemble_theory/forward_ensemble/)
- [Forward Ensemble Discussion](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175614)
- [Grad-CAM](https://reighns92.github.io/reighns-ml-blog/reighns_ml_journey/deep_learning/computer_vision/general/neural_network_interpretation/05_gradcam_and_variants/gradcam_explained/)
