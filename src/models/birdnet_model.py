import tensorflow as tf
import torch
from torch import nn
from src.models.custom_layers import ResidualBlock
from torchsummary import summary


# First BirdNet model


class BirdNet(nn.Module):
    """
    Defines the BirdNet neural network that classifies audios on 2 classes.
    Architecture :
        - Convolutional layer : 5*5 Conv + BN + ReLu
                                Maxpooling
        - Residual stack 1 : Downsampling block
                             2*ResidualBlock
        - Residual stack 2 : Downsampling block
                             2*ResidualBlock
        - Residual stack 3 : Downsampling block
                             2*ResidualBlock
        - Residual stack 4 : Downsampling block
                             2*ResidualBlock
        - Classification : 4*10 Conv + BN + ReLu + Dropout
                           1*1 Conv + BN + ReLu + Dropout
                           1*1 Conv + BN + Dropout
                           Global log-mean-exponential pooling
                           Sigmoid activation
    """

    def __init__(self):
        super(BirdNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.residual1 = self.make_residual_stack(32, 64, 2)
        self.residual2 = self.make_residual_stack(64, 128, 2)
        self.residual3 = self.make_residual_stack(128, 256, 2)
        self.residual4 = self.make_residual_stack(256, 512, 2)
        self.conv2 = nn.Conv2d(
            512, 512, kernel_size=(4, 10), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout(p=0.2)
        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1,
                               stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(1024)
        self.dropout2 = nn.Dropout(p=0.2)
        self.conv4 = nn.Conv2d(1024, 2, kernel_size=1,
                               stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.global_lme = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def make_residual_stack(self, in_channels: int, out_channels: int, n_blocks: int) -> nn.Module:
        layers = []
        # Downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        # Residual blocks
        for i in range(n_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        # Residual stacks
        out = self.residual1(out)
        out = self.residual2(out)
        out = self.residual3(out)
        out = self.residual4(out)
        out = self.relu(out)
        # Classification
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.dropout3(out)
        out = self.global_lme(out)
        out = self.sigmoid(out)
        return out

    def summary(self, input_shape):
        # summary(self, input_shape)

        x = torch.zeros((2, *input_shape))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print("Preprocessing    |   ", out.shape)
        out = self.maxpool(out)
        print("Maxpool          |   ", out.shape)
        # Residual stacks
        out = self.residual1(out)
        print("Res1             |   ", out.shape)
        out = self.residual2(out)
        print("Res2             |   ", out.shape)
        out = self.residual3(out)
        print("Res3             |   ", out.shape)
        out = self.residual4(out)
        print("Res4             |   ", out.shape)
        out = self.relu(out)
        # Classification
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        print("Classif 1st conv |   ", out.shape)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.dropout2(out)
        print("Classif 2nd conv |   ", out.shape)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.dropout3(out)
        print("Classif 3rd conv |   ", out.shape)
        out = self.global_lme(out)
        print("Global LME       |   ", out.shape)
        out = self.sigmoid(out)
        print("output shape     |   ", out.shape)


# loaded BirdNet keras model
class BirdNet_loaded(tf.keras.Model):
    """
    BirdNet model from keras saved_model
    """

    def __init__(self, path, num_outputs, in_shape=(144000)):
        super(BirdNet_loaded, self).__init__()
        self.in_shape = in_shape
        self.num_outputs = num_outputs
        self.path = path
        self.model = tf.keras.models.load_model(self.path)
        print(self.model.layers)
        self.new_model = tf.keras.Model(
            inputs=self.model.layers[0].input, outputs=self.model.layers[-2].output)

    def call(self, x, training=False):
        x = self.new_model(x, training=training)
        x = tf.keras.layers.Dense(self.num_outputs)(x, training=training)
        x = tf.keras.layers.Activation('sigmoid')(x, training=training)
        return x

    def summary(self):
        x = tf.keras.layers.Input(shape=self.in_shape)
        model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


if __name__ == "__main__":
    # Test
    # model = tf.keras.load_model(
    #     './src/models/saved_models/BirdNet_checkpoints')
    model = BirdNet_loaded(
        path='./src/models/saved_models/BirdNet_checkpoints', num_outputs=2)
    model.build(input_shape=(None, 144000))
    model.summary()
