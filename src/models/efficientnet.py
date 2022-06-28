import torch
from torch import nn
import warnings
from efficientnet_pytorch import EfficientNet as EfficientNet_Pytorch
from torchsummary import summary
warnings.filterwarnings('ignore')


# Define EfficientNet model

class EfficientNet(nn.Module):
    """
    EfficientNet model
    """

    def __init__(self, num_classes=2):
        """
        Initialize the model
        :param num_classes: number of classes
        """
        super(EfficientNet, self).__init__()
        self.model = EfficientNet_Pytorch.from_pretrained('efficientnet-b5')
        self.num_classes=num_classes
        feature = self.model._fc.in_features
        self.model._fc = nn.Sequential(nn.Linear(
            in_features=feature, out_features=num_classes, bias=True), nn.Softmax(1))

    def forward(self, x):
        """
        Forward pass
        :param x: input image
        :return: output
        """
        return self.model(x)


if __name__ == '__main__':
    model = EfficientNet()
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(device)
    summary(model, input_size=(3, 129, 129))
    model.to(device)
