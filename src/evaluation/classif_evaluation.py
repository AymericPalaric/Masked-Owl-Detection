import torch
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, recall_score, precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from src.models import efficientnet
from src.utils import torch_utils, transform_utils
import torchvision
from tqdm import tqdm


class ClassifEvaluator():
    """
    Evaluates a classification model.
    Attributes:
    -----------
        - model_class: base class of the model to be evaluated
        - device: device to use (see torch.cuda.device)
        - train_test: whether to evaluate on train or test dataset
        - batch_size: size of the batches to iterate over
        - name: basename for the saved files
        - model_path: path towards weights of the previously trained model
        - metrics: list of metrics to display in terminal lines
        - metrics_name: corresponding list of names
        - num_workers
        - pin_memory

    Methods:
    --------
        - get_preds(thresh=0.5, verbose=False): computes predictions over the given dataset
            Args:
                - thresh: threshold used for the classification prediction (if softmax is the last layer of the model)
                - verbose: whether to display indication on the computation status
            Returns:
                tuple(preds, labels, losses)

        - evaluate(): computes metrics over the predictions
            Returns:
                - metrics_dict: attribute created to store the pairs {metric_name: metric_value}

        - conf_matrix(): computes and save as png file the confusion matrix.
    """

    def __init__(self, model_class, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), model_path=None, train_test='test', batch_size=32, name=None, metrics=None, metrics_name=None, num_workers=4, pin_memory=None, **model_kwargs):
        """
        Initialize the evaluator.
        """

        self.model_class = model_class
        self.device = device
        self.train_test = train_test
        self.batch_size = batch_size
        self.name = name if name is not None else 'unnamed'
        self.model_path = model_path if model_path is not None else 'trained_models/' + name + '.pt'
        self.metrics = metrics
        self.metrics_name = metrics_name
        self.num_workers = num_workers
        self.pin_memory = pin_memory if pin_memory is not None else torch.cuda.is_available()

        self.model_kwargs = model_kwargs
        assert self.train_test in ['train', 'test']
        # Transforms
        self.reshape_size = (129, 129)
        audio_transform = transform_utils.baseline_transform

        image_transform_no_standardization = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), torchvision.transforms.Resize(self.reshape_size)])
        mean, std = torch_utils.compute_mean_std_classif(
            audio_transform=audio_transform, image_transform=image_transform_no_standardization, batch_size=batch_size, num_workers=num_workers)
        print(f"Mean dataset: {mean}, std dataset: {std}")
        image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(
        ), torchvision.transforms.Resize(self.reshape_size), torchvision.transforms.Normalize(mean=mean, std=std)])

        # Datasets
        self.dataset = torch_utils.create_classif_dataset(
            audio_transform, image_transform, train_test=True)

        # Dataloaders
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=True,
                                                      drop_last=True, num_workers=num_workers, pin_memory=self.pin_memory)

        # Load model weights
        self.model = self.model_class(**self.model_kwargs)
        self.model.load_state_dict(torch.load(
            self.model_path, map_location=torch.device(self.device)))
        self.model = self.model.to(self.device)

    def get_preds(self, thresh=None, verbose=True):
        """
        Get the predictions.
        """
        self.preds = []
        self.labels = []
        self.losses = []
        iterator = tqdm(self.dataloader) if verbose else self.dataloader
        for batch in iterator:
            with torch.no_grad():
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                preds = self.model(x)

                if thresh is not None:
                    preds[preds >= thresh] = 1
                    preds[preds < thresh] = 0

                preds = torch.argmax(preds, dim=1)

                self.preds.extend(preds.cpu().numpy())
                self.labels.extend(y.cpu().numpy())
                #self.losses.append(self.model.loss(preds, y).item())
        return (self.preds, self.labels, self.losses)

    def evaluate(self, thresh=None, verbose=True):
        """
        Evaluate the model.
        """

        self.model.eval()
        self.metrics = self.metrics if self.metrics is not None else [
            accuracy_score, recall_score, precision_score, lambda true, pred: precision_recall_fscore_support(true, pred, average='binary')]
        self.metrics_name = self.metrics_name if self.metrics_name is not None else [
            'accuracy', 'recall', 'precision', 'ALL']
        self.get_preds(thresh=thresh, verbose=verbose)

        self.metrics_dict = {}

        for metric, name in zip(self.metrics, self.metrics_name):
            self.metrics_dict[name] = metric(self.preds, self.labels)

        #self.metrics_dict['loss'] = np.mean(self.losses)

        return self.metrics_dict

    def conf_matrix(self, thresh=None, verbose=True):
        """
        Plot the confusion matrix.
        """
        preds, labels = self.get_preds(thresh=thresh, verbose=verbose)[:2]

        #conf_mx = confusion_matrix(labels, preds)

        # plt.matshow(conf_mx)
        ConfusionMatrixDisplay.from_predictions(labels, preds, normalize=None)
        plt.savefig("trained_models/conf_matrix_" +
                    self.name+f"_{int(100*thresh)}.png")


if __name__ == '__main__':
    evaluator = ClassifEvaluator(model_path='trained_models/efficientnet_model_9.pt', model_class=efficientnet.EfficientNet, train_test='test',
                                 batch_size=32, name='efficientnet', num_workers=1, pin_memory=False)
    print(evaluator.evaluate())
    evaluator.plot_conf_matrix()
