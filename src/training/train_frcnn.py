import torch
from torch.utils.data import DataLoader
from src.utils import path_utils
from src.utils.dataset import OneShotDataset,transform_audio
from src.models.faster_rcnn import rcnn_pretrained_backbone_train


def loss_faster_rcnn(dict_losses : dict, training_rpn : bool, training_head : bool, device):
    """ Reduce the dictionnary of losses

    Args:
        dict_losses (dict): Dictionnary of losses
        training_rpn (bool): Bool to train the rpn
        training_head (bool): Bool to train the head

    Returns:
        (torch.Tensor): Loss
    """
    # Dict("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg")
    loss = torch.zeros((), dtype=torch.float32, device=device)
    if training_rpn:
        loss += dict_losses["loss_objectness"]
        loss += dict_losses["loss_rpn_box_reg"]
    if training_head:
        loss += dict_losses["loss_classifier"]
        loss += dict_losses["loss_box_reg"]
    return loss

def average_losses(dataloader, mean_loss):
    mean_loss["loss_objectness"] /= len(dataloader)
    mean_loss["loss_rpn_box_reg"] /= len(dataloader)
    mean_loss["loss_classifier"] /= len(dataloader)
    mean_loss["loss_box_reg"] /= len(dataloader)
    return mean_loss

def accumulate_losses(mean_loss, losses):
    mean_loss["loss_objectness"] += losses["loss_objectness"].item()
    mean_loss["loss_rpn_box_reg"] += losses["loss_rpn_box_reg"].item()
    mean_loss["loss_classifier"] += losses["loss_classifier"].item()
    mean_loss["loss_box_reg"] += losses["loss_box_reg"].item()

def train_loop(dataloader,device, model, loss_fn, optimizer, scheduler, macro_batch=1):
    # Initialize training
    model.train()
    optimizer.zero_grad()

    mean_loss = {"loss_objectness": 0., "loss_rpn_box_reg": 0., "loss_classifier": 0., "loss_box_reg": 0.}
    # Iterate over the dataset
    for batch, (X, targets) in enumerate(dataloader):
        # Work with the GPU if available
        X = list(x.to(device) for x in X)
        targets = list({k: v.to(device) for k, v in t.items()} for t in targets)
        # Compute prediction error
        outputs = model(X, targets)
        losses, detection = outputs
        accumulate_losses(mean_loss, losses)
        loss = loss_fn(losses)
        print(loss)
        # Backpropagation
        loss.backward()
        if (batch+1) % macro_batch == 0 or batch == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        # Print metrics
        if batch % 30 == 0 or batch == len(dataloader) - 1:
            loss_value, current = loss.item(), (batch+1) * len(X)
            print(loss_value)
    return average_losses(dataloader, mean_loss)

def test_loop(dataloader, model, loss_fn, scheduler=None):
    """ Test function No metrics yet implemented """
    # Initialize training

    loss_value = 0.

    model.train()

    mean_loss = {"loss_objectness": 0., "loss_rpn_box_reg": 0., "loss_classifier": 0., "loss_box_reg": 0.}

    # Iterate over the dataset
    with torch.no_grad(): # Inference mode not working ?
        for batch, (X, targets) in enumerate(dataloader):

            # Work with the GPU if available
            X = list(x.to(device) for x in X)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            losses, detections = model(X, targets)
            accumulate_losses(mean_loss, losses)
            loss = loss_fn(losses)

            loss_value += loss.item()

    loss_value /= len(dataloader)


    mean_loss = average_losses(dataloader, mean_loss)

    if scheduler is not None:
        scheduler.step(total_loss(mean_loss))

    return mean_loss

def total_loss(losses):
    return losses["loss_objectness"] + losses["loss_rpn_box_reg"] + losses["loss_classifier"] + losses["loss_box_reg"]

def custom_collate(batch):
    return tuple(zip(*batch))

def mean_std(device):
    dataset = OneShotDataset(detection_dir=path_utils.get_detection_data_path(),
                                                    transform_audio=transform_audio, device=device,mean=0,std=1) 
    dataloader = torch.utils.data.DataLoader(dataset,collate_fn=custom_collate,batch_size=1)
    mean, std = 0., 0.
    for X, Y in dataloader:
        for x in X:
            mean += torch.mean(x)
            std += torch.std(x) ** 2
    mean /= len(dataloader)
    std /= len(dataloader)
    return mean, std

if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    mean,std=mean_std(device)
    print(f"mean: {mean}, std: {(std) ** 0.5}")

    detection_dataset = OneShotDataset(detection_dir=path_utils.get_detection_data_path(),
                                                    transform_audio=transform_audio, device=device,mean=mean,std=std)

    detection_dataloader = torch.utils.data.DataLoader(detection_dataset,collate_fn=custom_collate,batch_size=32)   

    model = rcnn_pretrained_backbone_train(num_classes=2,anchor_sizes=((32, 64,128,256),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    sgd_optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer,
                                                step_size=3,
                                                gamma=0.1)
    loss_fn_frrcnn=lambda x:loss_faster_rcnn(x,True,True)

    epochs=3
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loop(dataloader=detection_dataloader, model=model, loss_fn=loss_fn_frrcnn, optimizer=sgd_optimizer, scheduler=lr_scheduler, macro_batch=1)   
        print("---------------------------")
    print("Finished training")

                             