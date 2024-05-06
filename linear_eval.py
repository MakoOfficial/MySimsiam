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
# from datasets import get_dataset
from datasets.__init__Bone import get_dataset, get_single
from optimizers import get_optimizer, LR_Scheduler


def main(args):
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
    classifier = nn.Linear(in_features=model.output_dim, out_features=228, bias=True).to(args.device)
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
    optimizer = get_optimizer(
        args.eval.optimizer.name, classifier,
        lr=args.eval.base_lr * args.eval.batch_size / 256,
        momentum=args.eval.optimizer.momentum,
        weight_decay=args.eval.optimizer.weight_decay)

    # define lr scheduler
    lr_scheduler = LR_Scheduler(
        optimizer,
        args.eval.warmup_epochs, args.eval.warmup_lr * args.eval.batch_size / 256,
        args.eval.num_epochs, args.eval.base_lr * args.eval.batch_size / 256,
                                 args.eval.final_lr * args.eval.batch_size / 256,
        len(train_loader),
    )

    loss_meter = AverageMeter(name='Loss')
    acc_meter = AverageMeter(name='Accuracy')
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    # Start training
    global_progress = tqdm(range(0, args.eval.num_epochs), desc=f'Evaluating')
    for epoch in global_progress:
        loss_meter.reset()
        model.eval()
        classifier.train()
        local_progress = tqdm(train_loader, desc=f'Epoch {epoch}/{args.eval.num_epochs}', disable=True)

        total_size = 0.
        training_loss = 0.

        for idx, (images, labels) in enumerate(local_progress):
            classifier.zero_grad()
            with torch.no_grad():
                feature = model(images.to(args.device))

            labels = (labels-1).type(torch.LongTensor).to(args.device).squeeze()
            preds = classifier(feature)
            preds = preds.squeeze()

            loss = loss_fn(preds, labels)

            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=feature.shape[0])
            lr = lr_scheduler.step()

            training_loss += loss.item()
            total_size += feature.shape[0]

            local_progress.set_postfix({'lr': lr, 'loss': loss_meter.val, 'loss_avg': loss_meter.avg})
        train_avg = training_loss / total_size
        print(f"epoch {epoch+1}, avg_loss: {train_avg}")

    classifier.eval()
    valid_loss, total = 0., 0.
    acc_meter.reset()
    for idx, (images, labels) in enumerate(test_loader):
        with torch.no_grad():
            feature = model(images.to(args.device))
            preds = classifier(feature).argmax(dim=1)
            labels = (labels-1).type(torch.LongTensor)
            MAE_loss = F.l1_loss(preds, labels, reduction="sum").item()
            valid_loss += MAE_loss
            total += feature.shape[0]
            acc_meter.update(val=MAE_loss, n=feature.shape[0])
    print(f'Accuracy = {acc_meter.avg:.2f}, avg valid loss :{(valid_loss / total):.2f}')


if __name__ == "__main__":
    main(args=get_args())
