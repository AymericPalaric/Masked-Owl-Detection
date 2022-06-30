

import matplotlib.pyplot as plt
import torchvision
import torch
import json
import argparse
nn = torch.nn
from src.utils import path_utils
from src.utils.dataset import OneShotDataset,transform_audio
from src.models.faster_rcnn import rcnn_pretrained_backbone_train
from src.training.train_frcnn import *


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    # In windows, add: --model_args '{"key_1": "value_1", "key_2": "value_2"}'
    parser.add_argument("--model_args", required=False, type=json.loads,
                        help="Model kwargs", default='{}')


    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    n_workers = args.n_workers
    model_args = args.model_args
    lr = args.lr


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    mean,std=mean_std(device)
    print(f"mean: {mean}, std: {(std) ** 0.5}")


    detection_dataset = OneShotDataset(detection_dir=path_utils.get_detection_data_path(),
                                                    transform_audio=transform_audio, device=device,mean=mean,std=std)

    detection_dataloader = torch.utils.data.DataLoader(detection_dataset,collate_fn=custom_collate,batch_size=batch_size,num_workers=n_workers)   

    model = rcnn_pretrained_backbone_train(num_classes=2,anchor_sizes=((32, 64,128,256),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    sgd_optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(sgd_optimizer,
                                                step_size=3,
                                                gamma=0.1)

    loss_fn_frrcnn=lambda x:loss_faster_rcnn(x,True,True,device)

    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_loop(dataloader=detection_dataloader, device=device, model=model, loss_fn=loss_fn_frrcnn, optimizer=sgd_optimizer, scheduler=lr_scheduler, macro_batch=1)   
        print("---------------------------")
    print("Finished training")