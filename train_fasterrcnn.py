from __future__ import division, print_function

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.postprocess_fasterrcnn import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse
import numpy as np
import copy

import torch
import torch.nn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import datasets

from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--train", default='True', help="if True train the model (default=True).")
    parser.add_argument("--eval", default='False', help="if True evaluate the model on validation data (default=False)")
    parser.add_argument("--gpus", type=str, help="Id of the gpu(s) to use (only supports single GPU training for now).",
                        default="0")
    parser.add_argument("--no_cuda", help="Deactivate CUDA support", action="store_true")
    parser.add_argument("--iou_thresh", type=str, help="IoU threshold", default="0.5")
    parser.add_argument("--nms_thresh", type=str, help="NMS threshold", default="0.4")
    parser.add_argument("--conf_thresh", type=str, help="Confidence threshold", default="0.25")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs_fasterrcnn")

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    use_cuda = not opt.no_cuda
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    train = opt.train == 'True'
    valid = opt.eval == 'True'

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    backup_path = data_config["backup"]
    _, backup_name = os.path.split(opt.model_def)
    backup_name, _ = os.path.splitext(backup_name)

    # Get training configuration
    cfg = parse_model_config(opt.model_def)[0]
    batch_size = int(opt.batch_size) if opt.batch_size is not None else int(cfg['batch'])
    max_batches = int(cfg['max_batches'])
    subdivisions = int(cfg['subdivisions'])
    learning_rate = float(cfg['learning_rate'])
    momentum = float(cfg['momentum'])
    decay = float(cfg['decay'])
    steps = [float(step) for step in cfg['steps'].split(',')]
    scales = [float(scale) for scale in cfg['scales'].split(',')]

    # Augmentation from config file
    if not 'flip' in cfg.keys():
        cfg['flip'] = 0
    data_augmentation = dict()
    data_augmentation['jitter'] = float(cfg['jitter']) if 'jitter' in cfg.keys() else 0
    data_augmentation['hue'] = float(cfg['hue']) if 'hue' in cfg.keys() else 0
    data_augmentation['saturation'] = float(cfg['saturation']) if 'saturation' in cfg.keys() else 0
    data_augmentation['exposure'] = float(cfg['exposure']) if 'exposure' in cfg.keys() else 0
    data_augmentation['angle'] = float(cfg['angle']) if 'angle' in cfg.keys() else 0
    data_augmentation['flip'] = True if int(cfg['flip']) == 1 else False

    seed = int(time.time())
    eps = 1e-5

    # Test parameters
    conf_thresh = float(opt.conf_thresh)
    nms_thresh = float(opt.nms_thresh)
    iou_thresh = float(opt.iou_thresh)

    # Initiate model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # PRETRAINED ON COCO train 2017
    # Correct number of classes
    num_classes = len(class_names)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # print(model)

    # If the model is too big, we can change the backbone following section 2 in
    # https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=mTgWtixZTs3X

    # If specified we start from checkpoint
    if opt.pretrained_weights is not None:
        if opt.pretrained_weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.pretrained_weights, map_location=lambda storage, loc: storage))
            _, name = os.path.split(opt.pretrained_weights)
            load_epoch = int(name[:-4].split("_")[-1])
        else:
            load_epoch = 0
            print("Pretrained weights must end with .pth. Use pytorch pretrained weights.")
    else:
        load_epoch = 0

    # Get dataloader
    # With Faster R-CNN: no normalization, already in PyTorch model
    dataset = ListDatasetFasterRCNN(train_path, transform=transforms.ToTensor(), data_augmentation=data_augmentation)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    nsamples = dataset.Nsamples

    if valid:
        valid_dataset = ListDatasetFasterRCNN(valid_path, transform=transforms.ToTensor(), train=False)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )

    # optimizer = torch.optim.Adam(model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    images_seen = load_epoch*len(dataset)
    init_epoch = load_epoch + 1
    max_epoch = max(opt.epochs, int(max_batches * batch_size / len(dataset)))
    processed_batches = load_epoch/batch_size

    loss_logger = DictSaver()
    loss_save_path = os.path.join(backup_path, backup_name + "_loss.csv")

    # Adapted from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    model = model.to(device)
    for epoch in range(init_epoch, max_epoch):
        if train:
            model.train()
            start_time = time.time()

            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                batches_done = len(dataloader) * epoch + batch_i

                images = list(image.to(device) for image in imgs)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                optimizer.zero_grad()
                losses.backward()

                loss_logger.add_data(loss_dict)
                if batch_i % 10 == 0:  # print log every 10 batches
                    # ----------------
                    #   Log progress
                    # ----------------

                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))
                    log_str += "Learning rate={}\n".format(optimizer.state_dict()['param_groups'][0]['lr'])
                    log_str += "Momentum={}\n".format(optimizer.state_dict()['param_groups'][0]['momentum'])
                    log_str += "\n-- Losses --\n"
                    loss_str = [lk + ": " + str(lv.item()) for lk, lv in loss_dict.items()]
                    log_str += "\n".join(loss_str)
                    log_str += "\n-- Total loss: {}\n".format(losses.item())

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += "\n---- ETA {}".format(time_left)

                    print(log_str)

                optimizer.step()

        loss_logger.save(loss_save_path)
        if valid:
            if epoch % opt.evaluation_interval == 0:
                print("\n---- Evaluating Model ----\n")
                # Evaluate the model on the validation set
                model.eval()
                Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

                batch_metrics = []
                labels = []
                for eval_i, (_, imgs_eval, targets_eval) in enumerate(tqdm.tqdm(valid_dataloader, desc="Detecting objects")):
                    images = list(img.type(torch.cuda.FloatTensor if use_cuda else torch.FloatTensor) for img in imgs_eval)
                    labels += [t['labels'] for t in targets_eval]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets_eval]
                    with torch.no_grad():
                        outputs = model(images)  # List[Dict[Tensor]] with Dict containing 'boxes', 'labels' and 'scores'
                    outputs = postprocess(outputs, conf_thresh, nms_thresh)
                    outputs = [o.to(device) for o in outputs]

                    # Compute true positives, predicted scores and predicted labels per sample
                    batch_metrics += batch_statistics(outputs, targets, iou_threshold=iou_thresh)

                # Concatenate sample statistics
                if len(batch_metrics) == 0:
                    true_positives, pred_scores, pred_labels = np.array([]), np.array([]), np.array([])
                else:
                    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in
                                                                list(zip(*batch_metrics))]
                precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

                evaluation_metrics = [
                    ("val_precision", precision.mean()),
                    ("val_recall", recall.mean()),
                    ("val_mAP", AP.mean()),
                    ("val_f1", f1.mean()),
                ]
                logger.list_of_scalars_summary("Evaluation", evaluation_metrics, epoch)

                # Print class APs and mAP
                ap_table = [["Index", "Class name", "AP"]]
                for i, c in enumerate(ap_class):
                    ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
                print(AsciiTable(ap_table).table)
                print("---- mAP {}".format(AP.mean()))

                if not train:
                    break

        if train and epoch % opt.checkpoint_interval == 0:
            save_path = os.path.join(backup_path, backup_name + "_epoch_%d.weights" % epoch)
            torch.save(model.state_dict(), save_path)
            print("Model saved at %s" % save_path)

    logger.close()
    print("Normal ending of the program.")
