import torch
import torchvision
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import mobilenet_backbone
from torchvision.models.detection.anchor_utils import AnchorGenerator
nn = torch.nn



def rcnn_pretrained_backbone(num_classes : int, anchor_sizes : tuple, aspect_ratios : tuple, parameters : dict = {}):
    """
    Return a faster rcnn with a mobilenetv3 backbone.

    Args:
        num_classes (int): Number of classes expected to return (background should be taken into account)
        parameters (dict, optional): Dictionnary for the different following parameters. Defaults to {}.

    Returns:
        (torchvision.models.detection.faster_rcnn.FasterRCNN): Model implementation of pytorch

    """

    pretrained_backbone=True
    trainable_backbone_layers=6 # All backbone is trainable

    backbone = mobilenet_backbone("mobilenet_v3_large", pretrained_backbone, False, trainable_layers=trainable_backbone_layers)

    model = torchvision.models.detection.faster_rcnn.FasterRCNN(backbone,
                                                                num_classes,
                                                                rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
                                                                **parameters,)

    # Custom transform ie no transform (only postprocessing)
    model.transform = GeneralizedRCNNTransform(min_size=489, max_size=2000, image_mean=[0, 0, 0], image_std=[1, 1, 1])
    return model

def rcnn_pretrained_backbone_train(num_classes : int, anchor_sizes : tuple, aspect_ratios : tuple, parameters : dict = {}):
    """
    Return a jit compiled faster rcnn with a mobilenetv3 backbone.
    """
    model = rcnn_pretrained_backbone(num_classes, anchor_sizes, aspect_ratios, parameters)
    model._has_warned = True # Remove warning about "RCNN always returns a (Losses, Detections) tuple in scripting"
    return torch.jit.script(model)

