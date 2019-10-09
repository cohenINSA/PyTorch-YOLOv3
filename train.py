from __future__ import division

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils import evaluation as evaluation

from terminaltables import AsciiTable

import os
import time
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2, help="number of epochs")
    parser.add_argument("--batch_size", type=int, help="size of each image batch")
    parser.add_argument("--model_config", type=str, default="config/yolov3.cfg", help="path to training parameters file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--train", default='True', help="if True train the model (default=True).")
    parser.add_argument("--valid", default='True', help="if True evaluate the model on validation data (default=True)")
    parser.add_argument("--test", default='False', help="if True evaluate the model on test data (default=False)")
    parser.add_argument("--gpus", type=str, help="Id of the gpu(s) to use (only supports single GPU training for now).",
                        default="0")
    parser.add_argument("--no_cuda", help="Deactivate CUDA support", action="store_true")
    parser.add_argument("--iou_thresh", type=str, help="IoU threshold (default=0.5)", default="0.5")
    parser.add_argument("--nms_thresh", type=str, help="NMS threshold (default=0.4)", default="0.40")
    parser.add_argument("--conf_thresh", type=str, help="Confidence threshold (default=0.25)", default="0.25")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--tf_board", default="False", help="Log using tensorboard")
    opt = parser.parse_args()
    print(opt)

    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:" + opt.gpus)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    train = opt.train == 'True'
    valid = opt.valid == 'True'
    test = opt.test == 'True'

    if train and not valid and not test:
        print("Training YOLOv3 not available yet. Stopping execution here.")
        exit(1)

    tf_board = opt.tf_board == 'True'
    if tf_board:
        logger = Logger("logs_fasterrcnn")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    num_classes = len(class_names)
    backup_path = data_config["backup"]
    _, backup_name = os.path.split(opt.model_config)
    backup_name, _ = os.path.splitext(backup_name)

    # Get training configuration
    _, config_name = os.path.split(opt.model_config)
    cfg = parse_model_config(opt.model_config)[0]
    batch_size = int(opt.batch_size) if opt.batch_size is not None else int(cfg['batch'])
    max_batches = int(cfg['max_batches'])
    subdivisions = int(cfg['subdivisions'])
    learning_rate = float(cfg['learning_rate'])
    momentum = float(cfg['momentum'])
    decay = float(cfg['decay'])
    steps = [float(step) for step in cfg['steps'].split(',')]
    scales = [float(scale) for scale in cfg['scales'].split(',')]
    warmup = int(cfg['burn_in'])
    gradient_accumulation = batch_size/subdivisions

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
    model = Darknet(opt.model_config).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.weights is not None:
        if opt.weights.endswith(".pth"):
            model.load_state_dict(torch.load(opt.weights))
            # model.load_state_dict(torch.load(opt.pretrained_weights, map_location=lambda storage, loc: storage))
        else:
            model.load_darknet_weights(opt.weights)
        print("Loading weights from %s" % opt.weights)

    # Get dataloader
    dataset = ListDataset(train_path, img_size=opt.img_size, multiscale=opt.multiscale_training, train=True,
                          transform=transforms.ToTensor(), data_augmentation=data_augmentation)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )
    nsamples = dataset.Nsamples

    if valid:
        valid_dataset = ListDataset(valid_path, img_size=opt.img_size, multiscale=opt.multiscale_training, train=False,
                                    transform=transforms.ToTensor())
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=3,
            shuffle=False,
            num_workers=opt.n_cpu,
            pin_memory=True,
            collate_fn=valid_dataset.collate_fn,
        )

    # Why decay*batch size and learning rate/batch size:
    # https://github.com/AlexeyAB/darknet/issues/1943 (for darknet only)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate / batch_size, momentum=momentum,
                                weight_decay=decay * batch_size)
    init_epoch = int(model.seen / len(dataset))
    max_epoch = opt.epochs if opt.epochs is not None else int(max_batches * batch_size / len(dataset))
    processed_batches = int(model.seen / batch_size)
    optimizer.zero_grad()

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    # Loggers
    loss_logger = DictSaver()
    loss_save_path = os.path.join(backup_path, backup_name + "_loss.csv")
    map_logger = DictSaver()
    map_save_path = os.path.join(backup_path, backup_name + "_map.csv")
    iou_logger = DictSaver()
    iou_save_path = os.path.join(backup_path, backup_name + "_iou.csv")
    avg_loss_logger = DictSaver()
    avg_loss_save_path = os.path.join(backup_path, backup_name + "_avg_loss_epoch.csv")

    for epoch in range(init_epoch, max_epoch):
        if train:
            model.train()
            start_time = time.time()
            processed_batches = model.seen/batch_size
            lr = adjust_learning_rate_darknet(optimizer, processed_batches, cfg)
            for batch_i, (_, imgs, targets) in enumerate(dataloader):
                current_lr = adjust_learning_rate_darknet(optimizer, processed_batches, cfg)
                processed_batches += 1
                batches_done = len(dataloader) * epoch + batch_i

                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)

                loss, outputs = model(imgs, targets)
                loss.backward()

                if batches_done % gradient_accumulation:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                if batch_i % 10 == 0:
                    # ----------------
                    #   Log progress
                    # ----------------

                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

                    metric_table = [["Metrics", *["YOLO Layer {}".format(i) for i in range(len(model.yolo_layers))]]]

                    # Log metrics at each YOLO layer
                    for i, metric in enumerate(metrics):
                        formats = {m: "%.6f" for m in metrics}
                        formats["grid_size"] = "%2d"
                        formats["cls_acc"] = "%.2f%%"
                        row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                        metric_table += [[metric, *row_metrics]]

                        # Tensorboard logging
                        tensorboard_log = []
                        for j, yolo in enumerate(model.yolo_layers):
                            for name, metric in yolo.metrics.items():
                                if name != "grid_size":
                                    tensorboard_log += [("{}_{}".format(name, j+1), metric)]
                        tensorboard_log += [("loss", loss.item())]
                        logger.list_of_scalars_summary("training", tensorboard_log, batches_done)

                    log_str += AsciiTable(metric_table).table
                    log_str += "\nTotal loss {}".format(loss.item())

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += "\n---- ETA {}".format(time_left)

                    print(log_str)

            model.seen = len(dataloader.dataset) * (epoch+1)

            if train and epoch % opt.checkpoint_interval == 0:
                save_path = os.path.join(backup_path, backup_name + "_epoch_%d.weights" % epoch)
                torch.save(model.state_dict(), save_path)
                print("Model saved at %s" % save_path)

        if valid:
            if epoch % opt.evaluation_interval == 0:
                print("\n---- Evaluating Model ----")
                # Evaluate the model on the validation set
                model.eval()
                # Lists to store detected and true boxes, labels, scores
                det_boxes = list()
                det_labels = list()
                det_scores = list()
                true_boxes = list()
                true_labels = list()

                with torch.no_grad():
                    for eval_i, (imgs_paths, imgs_eval, targets_eval) in enumerate(tqdm.tqdm(valid_dataloader, desc="Detecting objects")):
                        images = imgs_eval.to(device)
                        try:
                            outputs = model(images)  # Tensor of shape (Nbatch, Ndet, 5+num_classes)
                        except RuntimeError as e:
                            print("Runtime Error with image(s) ", imgs_paths)
                            print("Error: {}".format(e))
                            # outputs = list()
                            # outputs.extend([{'boxes': torch.empty((0, 4), dtype=torch.float64),
                            #                  'labels': torch.empty(0, dtype=torch.int64),
                            #                  'scores': torch.empty(0)}] * len(images))
                            outputs = torch.empty((len(imgs_eval), 0, 5+num_classes))

                        outputs = outputs.to(device)
                        det_boxes_batch, det_labels_batch, det_scores_batch = \
                            evaluation.postproces_batch_yolo(outputs, conf_thresh, nms_thresh, num_classes, device)

                        #print("\nTargets eval=", targets_eval)

                        targets = list()
                        for image_i in range(len(imgs_eval)):
                            targets_image = targets_eval[targets_eval[:, 0] == image_i]
                            target = dict()

                            boxes_image = targets_image[:, 2:]
                            boxes_image = xywh2xyxy(boxes_image) * opt.img_size
                            boxes_image = np.clip(boxes_image, 0, opt.img_size)
                            target['boxes'] = boxes_image

                            label_images = targets_image[:, 1]
                            target['labels'] = label_images
                            targets.append(target)
                        #print("TARGETS reconstructed = ", targets)

                        boxes = [t['boxes'].to(device) for t in targets]
                        labels = [t['labels'].to(device) for t in targets]

                        # Store GT for mAP calculation
                        det_boxes.extend(det_boxes_batch)
                        det_labels.extend(det_labels_batch)
                        det_scores.extend(det_scores_batch)
                        true_boxes.extend(boxes)
                        true_labels.extend(labels)

                    # Compute mAP
                    APs, mAP, IoU = evaluation.compute_map(det_boxes, det_labels, det_scores, true_boxes,
                                                           true_labels, num_classes, device, iou_thresh, bkgd=False)
                    # Print class APs and mAP
                    APs = APs.squeeze().tolist()
                    ap_table = [["Index", "Class name", "AP"]]
                    for i, c in enumerate(APs):
                        ap_table += [[i, class_names[i], "%.5f" % c]]
                    print(AsciiTable(ap_table).table)
                    print("---- mAP at IoU thresh {}: {}".format(iou_thresh, mAP))
                    print("---- IoU at conf_thresh {}: {}".format(conf_thresh, IoU))
                    map_logger.add_data({'mAP': [mAP]}, epoch)
                    iou_logger.add_data({'IoU': [IoU]}, epoch)

                    if not train:
                        break

                    map_logger.save(map_save_path)
                    iou_logger.save(iou_save_path)

    if test:
        print("\n---- Testing Model ----\n")
        # Testing the model on the validation set
        model.eval()
        # Lists to store detected and true boxes, labels, scores
        det_boxes = list()
        det_labels = list()
        det_scores = list()
        true_boxes = list()
        true_labels = list()

        with torch.no_grad():
            for eval_i, (imgs_paths, imgs_eval, targets_eval) in enumerate(
                    tqdm.tqdm(valid_dataloader, desc="Detecting objects")):
                images = imgs_eval.to(device)
                try:
                    outputs = model(images)  # Tensor of shape (Nbatch, Ndet, 5+num_classes)
                except RuntimeError as e:
                    print("Runtime Error with image(s) ", imgs_paths)
                    print("Error: {}".format(e))
                    # outputs = list()
                    # outputs.extend([{'boxes': torch.empty((0, 4), dtype=torch.float64),
                    #                  'labels': torch.empty(0, dtype=torch.int64),
                    #                  'scores': torch.empty(0)}] * len(images))
                    outputs = torch.empty((len(imgs_eval), 0, 5 + num_classes))

                outputs = outputs.to(device)
                det_boxes_batch, det_labels_batch, det_scores_batch = \
                    evaluation.postproces_batch_yolo(outputs, conf_thresh, nms_thresh, num_classes, device)

                # print("\nTargets eval=", targets_eval)

                targets = list()
                for image_i in range(len(imgs_eval)):
                    targets_image = targets_eval[targets_eval[:, 0] == image_i]
                    target = dict()

                    boxes_image = targets_image[:, 2:]
                    boxes_image = xywh2xyxy(boxes_image) * opt.img_size
                    boxes_image = np.clip(boxes_image, 0, opt.img_size)
                    target['boxes'] = boxes_image

                    label_images = targets_image[:, 1]
                    target['labels'] = label_images
                    targets.append(target)
                # print("TARGETS reconstructed = ", targets)

                boxes = [t['boxes'].to(device) for t in targets]
                labels = [t['labels'].to(device) for t in targets]

                # Store GT for mAP calculation
                det_boxes.extend(det_boxes_batch)
                det_labels.extend(det_labels_batch)
                det_scores.extend(det_scores_batch)
                true_boxes.extend(boxes)
                true_labels.extend(labels)

            # Compute mAP
            APs, mAP, IoU = evaluation.compute_map(det_boxes, det_labels, det_scores, true_boxes,
                                                   true_labels, num_classes, device, iou_thresh, bkgd=False)
            # Print class APs and mAP
            APs = APs.squeeze().tolist()
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(APs):
                ap_table += [[i, class_names[i], "%.5f" % c]]
            print(AsciiTable(ap_table).table)
            print("---- mAP at IoU thresh {}: {}".format(iou_thresh, mAP))
            print("---- IoU at conf_thresh {}: {}".format(conf_thresh, IoU))

    if tf_board:
        logger.close()
    print("Normal ending of the program.")
