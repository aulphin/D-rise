# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Defines wrappers for vision-based models."""

import logging
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from constants import Device, ModelTask

module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)

try:
    import torch
    import torch.nn as nn
    from torch import Tensor
except ImportError:
    Tensor = Any
    module_logger.debug('Could not import torch, required if using a '
                        'PyTorch model')
try:
    import common as od_common
    from common import \
        GeneralObjectDetectionModelWrapper
except ImportError:
    GeneralObjectDetectionModelWrapper = object
    module_logger.debug('Could not import vision_explanation_methods, ' +
                        'required if using PytorchDRiseWrapper')


try:
    import torchvision
    from torchvision import transforms as T
except ImportError:
    module_logger.debug('Could not import torchvision, required if '
                        'using PytorchDRiseWrapper')

try:
    from mlflow.pyfunc import PyFuncModel
except ImportError:
    PyFuncModel = Any
    module_logger.debug('Could not import mlflow, required if using an '
                        'mlflow model')

FASTAI_MODEL_SUFFIX = "fastai.learner.Learner'>"
BOXES = 'boxes'
LABELS = 'labels'
SCORES = 'scores'
COLON = ":"




def _filter_score(orig_prediction: dict, score_threshold: float = 0.5):
    """Filter out predictions with confidence scores < score_threshold.

    :param orig_prediction: Original model prediction
    :type orig_prediction: dict
    :param score_threshold: Score threshold to filter by
    :type score_threshold: float
    :return: Model predictions filtered out by score_threshold
    :rtype: dict
    """
    keep = orig_prediction[SCORES] > score_threshold

    filter_prediction = orig_prediction
    filter_prediction[BOXES] = filter_prediction[BOXES][keep]
    filter_prediction[SCORES] = filter_prediction[SCORES][keep]
    filter_prediction[LABELS] = filter_prediction[LABELS][keep]
    return filter_prediction


def _apply_nms(orig_prediction: dict, iou_threshold: float = 0.5):
    """Perform nms on the predictions based on their IoU.

    :param orig_prediction: Original model prediction
    :type orig_prediction: dict
    :param iou_threshold: iou_threshold for nms
    :type iou_threshold: float
    :return: Model prediction after nms is applied
    :rtype: dict
    """
    keep = torchvision.ops.nms(orig_prediction[BOXES],
                               orig_prediction[SCORES],
                               iou_threshold)

    nms_prediction = orig_prediction
    nms_prediction[BOXES] = nms_prediction[BOXES][keep]
    nms_prediction[SCORES] = nms_prediction[SCORES][keep]
    nms_prediction[LABELS] = nms_prediction[LABELS][keep]
    return nms_prediction


def _process_automl_detections_to_raw_detections(
        image_detections,
        label_dict: Dict[str, int],
        image_size: Tuple[int, int]) -> Dict[str, Tensor]:
    """Process AutoML mlflow object detections from a single image.

    The length of image_detections list will be equal to the number
    of objects the AutoML MLflow model detected in a single image.
    Below is an example of image_detections when the model detected
    two objects:

    [{'box': {'topX': 0.645,
              'topY': 0.373,
              'bottomX': 0.8276,
              'bottomY': 0.6102
                },
      'label': 'can',
      'score': 0.945},
     {'box': {'topX': 0.324,
              'topY': 0.5643,
              'bottomX': 0.6511,
              'bottomY': 0.865
                },
      'label': 'milk_bottle',
      'score': 0.93}
    ]
    The bbox coordinates need to be scaled by the image dimensions.

    :param image_detections: AutoML mlflow detections
    :type image_detections: list
    :param label_dict: Label dictionary mapping class names to an index
    :type label_dict: dict
    :param image_size: Image size
    :type image_size: tuple(int, int)
    :return: Raw detections
    :rtype: dict
    """

    x, y = image_size

    boxes = []
    scores = []
    labels = []
    for detection in image_detections:
        # predicted class label
        label = detection['label']
        # confidence score from model
        score = detection['score']
        # predicted bbox coordinates
        box = detection["box"]

        ymin, xmin, ymax, xmax = (
            box["topY"], box["topX"], box["bottomY"], box["bottomX"])

        # the automl mlflow model generates normalized bbox coordinates
        # they need to be scaled by the size of the image
        x_min_scaled, y_min_scaled = x * xmin, y * ymin
        x_max_scaled, y_max_scaled = x * xmax, y * ymax

        scores.append(score)
        labels.append(label)
        boxes.append([x_min_scaled, y_min_scaled, x_max_scaled, y_max_scaled])
    try:
        labels = [int(x) for x in labels]
    except BaseException:
        labels = [label_dict[x] for x in labels]

    return {
        "boxes": Tensor(boxes),
        "labels": Tensor(labels),
        "scores": Tensor(scores)}


def expand_class_scores(
        scores: Tensor,
        labels: Tensor,
        number_of_classes: int,
) -> Tensor:
    """Extrapolate a full set of class scores.

    Many object detection models don't return a full set of class scores, but
    rather just a score for the predicted class. This is a helper function
    that approximates a full set of class scores by dividing the difference
    between 1.0 and the predicted class score among the remaning classes.

    :param scores: Set of class specific scores. Shape [D] where D is number
        of detections
    :type scores: torch.Tensor
    :param labels: Set of label indices corresponding to predicted class.
        Shape [D] where D is number of detections
    :type labels: torch.Tensor (ints)
    :param number_of_classes: Number of classes model predicts
    :type number_of_classes: int
    :return: A set of expanded scores, of shape [D, C], where C is number of
        classes
    :type: torch.Tensor
    """
    number_of_detections = scores.shape[0]

    expanded_scores = torch.ones(number_of_detections, number_of_classes + 1)

    for i, (score, label) in enumerate(zip(scores, labels)):

        residual = (1. - score.item()) / (number_of_classes)
        expanded_scores[i, :] *= residual
        expanded_scores[i, int(label.item())] = score

    return expanded_scores


def _wrap_image_model(model, examples, model_task, is_function,
                      number_of_classes: int = None,
                      classes: Union[list, np.array] = None,
                      device=Device.AUTO.value):
    """If needed, wraps the model or function in a common API.

    Wraps the model based on model task and prediction function contract.

    :param model: The model or function to evaluate on the examples.
    :type model: function or model to wrap
    :param examples: The model evaluation examples.
    :type examples: ml_wrappers.DatasetWrapper
    or numpy.ndarray or pandas.DataFrame or panads.Series
    or scipy.sparse.csr_matrix or shap.DenseData
    or torch.Tensor.
    :param model_task: Parameter to specify whether the model is an
    'image_classification' or another type of image model.
    :type model_task: str
    :param classes: optional parameter specifying a list of class names
    the dataset
    :type classes: list or np.ndarray
    :param number_of_classes: optional parameter specifying the
    number of classes in the dataset
    :type number_of_classes: int
    :param device: optional parameter specifying the device to move the model
        to.
    :type device: str
    :return: The function chosen from given model and chosen domain, or
    model wrapping the function and chosen domain.
    :rtype: (function, str) or (model, str)
    """
    device = _get_device(device)
    _wrapped_model = model
    if model_task == ModelTask.IMAGE_CLASSIFICATION:
        try:
            if isinstance(model, nn.Module):
                model = WrappedPytorchModel(model, image_to_tensor=True)
                if not isinstance(examples, DatasetWrapper):
                    examples = DatasetWrapper(examples)
                eval_function, eval_ml_domain = _eval_model(
                    model, examples, model_task)
                return (
                    WrappedClassificationModel(model, eval_function, examples),
                    eval_ml_domain,
                )
        except (NameError, AttributeError):
            module_logger.debug(
                'Could not import torch, required if using a pytorch model'
            )

        if _is_fastai_model(model):
            _wrapped_model = WrappedFastAIImageClassificationModel(model)
        elif hasattr(model, '_model_impl'):
            if str(type(model._model_impl.python_model)).endswith(
                ("azureml.automl.dnn.vision.common.mlflow."
                    "mlflow_model_wrapper.MLFlowImagesModelWrapper'>")
            ):
                _wrapped_model = WrappedMlflowAutomlImagesClassificationModel(
                    model)
        elif _is_transformers_pipeline(model) or _is_callable_pipeline(model):
            _wrapped_model = WrappedTransformerImageClassificationModel(model)
    elif model_task == ModelTask.MULTILABEL_IMAGE_CLASSIFICATION:
        if _is_fastai_model(model):
            _wrapped_model = WrappedFastAIImageClassificationModel(
                model, multilabel=True
            )
    elif model_task == ModelTask.OBJECT_DETECTION:
        if hasattr(model, '_model_impl'):
            if str(type(model._model_impl.python_model)).endswith(
                ("azureml.automl.dnn.vision.common.mlflow."
                    "mlflow_model_wrapper.MLFlowImagesModelWrapper'>")
            ):
                _wrapped_model = WrappedMlflowAutomlObjectDetectionModel(
                    model, classes)
        elif _is_transformers_pipeline(model) or _is_callable_pipeline(model):
            _wrapped_model = WrappedObjectDetectionModel(
                model, number_of_classes, device)
    return _wrapped_model, model_task


def _get_device(device: str) -> str:
    """Sets the device to run computations on to the desired value.

    If device were set to "auto", then the desired device will be gpu (CUDA)
    if available. Otherwise, the device should be set to cpu.

    :param device: parameter specifying the device to move the model
        to.
    :type device: str
    :return: selected device to run computations on
    :rtype: str
    """
    if (device in [member.value for member in Device] or type(device) is int or device.isdigit() or device is None):
        if device == Device.AUTO.value:
            if torch.cuda.is_available():
                return Device.CUDA.value
            else:
                return Device.CPU.value
        return device
    elif COLON in device:
        split_vals = device.split(COLON)
        if len(split_vals) == 2:
            return COLON.join([_get_device(
                split_val) for split_val in split_vals])
    raise ValueError("Selected device is invalid")



class PytorchDRiseWrapper(GeneralObjectDetectionModelWrapper):
    """Wraps a PytorchFasterRCNN model with a predict API function.

    To be compatible with the D-RISE explainability method,
    all models must be wrapped to have the same output and input class and a
    predict function for object detection. This wrapper is customized for the
    FasterRCNN model from Pytorch, and can also be used with the RetinaNet or
    any other models with the same output class.
    """

    def __init__(self, model, number_of_classes: int,
                 device=Device.AUTO.value,
                 transforms=None,
                 iou_threshold=None,
                 score_threshold=None):
        """Initialize the PytorchDRiseWrapper.

        :param model: Object detection model
        :type model: PytorchFasterRCNN model
        :param number_of_classes: Number of classes the model is predicting
        :type number_of_classes: int
        :param device: Optional parameter specifying the device to move the
            model to. If not specified, then cpu is the default
        :type device: str
        :param transforms: Optional parameter specifying the transforms to
            apply to the image before passing it to the model
        :type transforms: torchvision.transforms
        :param iou_threshold: Optional parameter specifying the iou_threshold
            for nms. If not specified, the iou_threshold on the predict
            method is used.
        :type iou_threshold: float
        :param score_threshold: Optional parameter specifying the
            score_threshold for filtering detections. If not
            specified, the score_threshold on the predict method is
            used.
        :type score_threshold: float
        """
        self.transforms = transforms
        self._device = torch.device(_get_device(device))
        model.to(self._device)
        model.eval()

        self._model = model
        self._number_of_classes = number_of_classes
        self._iou_threshold = iou_threshold
        self._score_threshold = score_threshold

    def predict(self, x: Tensor, iou_threshold: float = 0.5,
                score_threshold: float = 0.5):
        """Create a list of detection records from the image predictions.

        :param x: Tensor of the image
        :type x: torch.Tensor
        :param iou_threshold: Intersection-over-Union (IoU) threshold for NMS (or
            the amount of acceptable error). Objects with error
            scores higher than the threshold will be removed.
        :type iou_threshold: float
        :param score_threshold: Threshold to filter detections based on
                            predicted confidence scores.
        :type score_threshold: float
        :return: Baseline detections to get saliency maps for
        :rtype: List of Detection Records
        """
        if self._iou_threshold is not None:
            iou_threshold = self._iou_threshold
        if self._score_threshold is not None:
            score_threshold = self._score_threshold
        with torch.no_grad():
            raw_detections = self._model(x)

            detections = []
            for raw_detection in raw_detections:
                raw_detection = _apply_nms(raw_detection, iou_threshold)

                # Note that FasterRCNN doesn't return a score for each
                # class, only the predicted class. DRISE requires a
                # score for each class.
                # We approximate the score for each class
                # by dividing (class score) evenly among the other classes.

                raw_detection = _filter_score(raw_detection, score_threshold)
                expanded_class_scores = expand_class_scores(
                    raw_detection[SCORES],
                    raw_detection[LABELS],
                    self._number_of_classes)

                detections.append(
                    od_common.DetectionRecord(
                        bounding_boxes=raw_detection[BOXES],
                        class_scores=expanded_class_scores,
                        objectness_scores=torch.tensor(
                            [1.0] * raw_detection[BOXES].shape[0]),
                    )
                )

            return detections


class MLflowDRiseWrapper():
    """Wraps a Mlflow model with a predict API function.

    To be compatible with the D-RISE explainability method,
    all models must be wrapped to have the same output and input class and a
    predict function for object detection. This wrapper is customized for the
    FasterRCNN model from AutoML. Unlike the Pytorch wrapper, this wrapper
    does not inherit from GeneralObjectDetectionModelWrapper as this super
    class requires predict to take a tensor input.
    """

    def __init__(self, model: PyFuncModel,
                 classes: Union[list, np.ndarray]) -> None:
        """Initialize the MLflowDRiseWrapper.

        :param model: mlflow model
        :type model: mlflow.pyfunc.PyFuncModel
        :param number_of_classes: Number of classes the model is predicting
        :type number_of_classes: int
        """

        self._model = model
        self._classes = classes
        self._number_of_classes = len(classes)
        self._label_dict = {label: (i + 1) for i, label in enumerate(classes)}

    def _mlflow_predict(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Perform the inference using the wrapped MLflow model.

        :param dataset: The dataset to predict on.
        :type dataset: pandas.DataFrame
        :return: The predicted data.
        :rtype: pandas.DataFrame
        """
        predictions = self._model.predict(dataset)
        return predictions

    def predict(self, dataset: pd.DataFrame, iou_threshold: float = 0.25,
                score_threshold: float = 0.5):
        """Predict the output value using the wrapped MLflow model.

        :param dataset: The dataset to predict on.
        :type dataset: pandas.DataFrame
        :param iou_threshold: Intersection-over-Union (IoU) threshold for NMS (or
            the amount of acceptable error). Objects with error
            scores higher than the threshold will be removed.
        :type iou_threshold: float
        :param score_threshold: Threshold to filter detections based on
                            predicted confidence scores.
        :type score_threshold: float
        :return: The predicted values.
        :rtype: numpy.ndarray
        """
        image_sizes = dataset['image_size']

        dataset = dataset.drop(['image_size'], axis=1)

        predictions = self._mlflow_predict(dataset)
        if not len(predictions['boxes']) == len(image_sizes):
            raise ValueError("Internal Error: Number of predictions "
                             "does not match number of images")

        if not len(predictions['boxes']) == 1:
            raise ValueError(
                "Currently, only 1 image can be passed to predict")

        detections = []
        for image_detections, img_size in \
                zip(predictions['boxes'], image_sizes):

            raw_detections = _process_automl_detections_to_raw_detections(
                image_detections, self._label_dict, img_size)

            # TODO: check if this is needed
            # No detections found - most likely in masked image
            if raw_detections[BOXES].nelement() == 0:
                detections.append(None)
                continue

            raw_detections = _apply_nms(raw_detections, iou_threshold)
            raw_detections = _filter_score(raw_detections, score_threshold)

            # Note that FasterRCNN doesn't return a score for each class, only
            # the predicted class. DRISE requires a score for each class.
            # We approximate the score for each class
            # by dividing (class score) evenly among the other classes.

            expanded_class_scores = expand_class_scores(
                raw_detections[SCORES],
                raw_detections[LABELS],
                self._number_of_classes)

            detections.append(
                od_common.DetectionRecord(
                    bounding_boxes=raw_detections[BOXES],
                    class_scores=expanded_class_scores,
                    objectness_scores=torch.tensor(
                        [1.0] * raw_detections[BOXES].shape[0]),
                ))

        return detections
