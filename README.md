### Dependencies
```
pip install -r requirements.txt
```

### Run SimSiam
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress

python main.py --data_dir ../archive/masked_1K_fold/fold_1 --log_dir ./logs/ -c configs/simsiam_bone.yaml --ckpt_dir ~/.cache/ --hide_progress
```

```
Training: 100%|#################################################################| 800/800 [11:46:06<00:00, 52.96s/it, epoch=799, accuracy=90.3]
Model saved to /root/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth
Evaluating: 100%|##########################################################################################################| 100/100 [08:29<00:00,  5.10s/it]
Accuracy = 90.83
Log file has been saved to ../logs/completed-simsiam-cifar10-experiment-resnet18_cifar_variant1(2)
```

To evaluate separately:
```
CUDA_VISIBLE_DEVICES=4 python linear_eval.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar_eval.yaml --ckpt_dir ~/.cache/ --hide_progress --eval_from ~/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth

creating file ../logs/in-progress_0111061045_simsiam-cifar10-experiment-resnet18_cifar_variant1
Evaluating: 100%|##########################################################################################################| 200/200 [16:52<00:00,  5.06s/it]
Accuracy = 90.87
```
![simsiam-cifar10-800e](simsiam-800e90.83acc.svg)

>`export DATA="/path/to/your/datasets/"` and `export LOG="/path/to/your/log/"` will save you the trouble of entering the folder name every single time!
