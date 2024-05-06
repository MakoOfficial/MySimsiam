import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from arguments import get_args
from augmentations import get_aug
from models import get_model, get_backbone
from tools import AverageMeter
from datasets.__init__Bone import get_single
from optimizers import get_optimizer, LR_Scheduler
from torch.optim.lr_scheduler import StepLR


def main(args, lr=None):
    train_loader = torch.utils.data.DataLoader(
        dataset=get_single(
            data_dir=args.data_dir,
            train=True
        ),
        batch_size=args.eval.batch_size,
        shuffle=True,
        **args.dataloader_kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=get_single(
            data_dir=args.data_dir,
            train=False
        ),
        batch_size=args.eval.batch_size,
        shuffle=False,
        **args.dataloader_kwargs
    )

    model = get_backbone(args.model.backbone)
    classifier = nn.Sequential(
        nn.Linear(in_features=model.output_dim, out_features=1024, bias=True),
        nn.ReLU(),
        nn.Linear(in_features=1024, out_features=228, bias=True)
    )

    print(f"classifier input dim is {model.output_dim}, output dim is 228")
    assert args.eval_from is not None
    save_dict = torch.load(args.eval_from, map_location='cpu')
    msg = model.load_state_dict({k[9:]: v for k, v in save_dict['state_dict'].items() if k.startswith('backbone.')},
                                strict=True)

    print(msg)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    # if torch.cuda.device_count() > 1: classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
    classifier = torch.nn.DataParallel(classifier)
    # define optimizer
    # optimizer = get_optimizer(
    #     args.eval.optimizer.name, classifier,
    #     lr=args.eval.base_lr * args.eval.batch_size / 256,
    #     momentum=args.eval.optimizer.momentum,
    #     weight_decay=args.eval.optimizer.weight_decay)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=wd)

    # define lr scheduler
    # lr_scheduler = LR_Scheduler(
    #     optimizer,
    #     args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
    #     args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
    #                              args.eval.final_lr * args.eval.batch_size / 256,
    #     len(train_loader),
    # )
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    best_loss = float('inf')
    for epoch in global_progress:
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)

        total_size = 0.
        training_loss = 0.

        for idx, (images, labels) in enumerate(local_progress):
            # classifier.zero_grad()
            optimizer.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            labels = (labels-1).type(torch.LongTensor).to(args.device).squeeze()
            print(f"labels.shape is {labels.shape}")
            preds = classifier(feature)
            preds = preds.squeeze()
            print(f"preds.shape is {preds.shape}")
            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            total_size += feature.shape[0]

        train_avg = training_loss / total_size

        classifier.eval()
        valid_loss, total = 0., 0.
        for idx, (images, labels) in enumerate(test_loader):
            with torch.no_grad():
                feature = model(images.to(args.device))
                output = classifier(feature)
                preds = torch.argmax(output, dim=1) + 1
                labels = labels.to(args.device)

                preds = preds.squeeze()
                labels = labels.squeeze()

                MAE_loss = F.l1_loss(preds, labels, reduction="sum").item()
                valid_loss += MAE_loss
                total += feature.shape[0]
        valid_avg = valid_loss / total
        print(f"epoch {epoch + 1}, train avg_loss: {train_avg}, valid avg_loss: {valid_avg}")
        if valid_avg < best_loss:
            best_loss = valid_loss
        lr_scheduler.step()
    print(f'best loss: {best_loss}')


if __name__ == "__main__":
    lr = 5e-4
    wd = 0
    main(args=get_args(), lr=lr)
