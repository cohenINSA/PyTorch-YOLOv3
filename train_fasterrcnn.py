from __future__ import division, print_function

from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils.postprocess_fasterrcnn import *
from utils import evaluation as evaluation

from terminaltables import AsciiTable

import os
import time
import datetime
import argparse
import numpy as np

from torch.utils.data import DataLoader
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision import transforms
import torch.optim as optim


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="number of epochs")
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
    parser.add_argument("--nms_thresh", type=str, help="NMS threshold (default=0.4)", default="0.4")
    parser.add_argument("--conf_thresh", type=str, help="Confidence threshold (default=0.25)", default="0.25")
    parser.add_argument("--freeze", default=None, type=int, help="After which layer stop to freeze the backbone (1 to 4).")
    parser.add_argument("--pretrained_database", default="ImageNet", help="Pretraining of the backbone: ImageNet or COCO")
    parser.add_argument("--tf_board", default="False", help="Log using tensorboard")
    opt = parser.parse_args()
    print(opt)

    use_cuda = not opt.no_cuda and torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:"+opt.gpus)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    train = opt.train == 'True'
    valid = opt.valid == 'True'
    test = opt.test == 'True'

    tf_board = opt.tf_board == 'True'
    if tf_board:
        logger = Logger("logs_fasterrcnn")

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    full_class_names = ['background']
    full_class_names.extend(class_names)
    backup_path = data_config["backup"]
    _, backup_name = os.path.split(opt.model_config)
    backup_name, _ = os.path.splitext(backup_name)

    # Get training configuration
    cfg = parse_model_config(opt.model_config)[0]
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
    if opt.pretrained_database == "COCO":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)  # PRETRAINED ON COCO train 2017
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)  # backbone pretrained on ImageNet

    # Correct number of classes
    num_classes = len(full_class_names)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # If the model is too big, we can change the backbone following section 2 in
    # https://colab.research.google.com/github/pytorch/vision/blob/temp-tutorial/tutorials/torchvision_finetuning_instance_segmentation.ipynb#scrollTo=mTgWtixZTs3X

    # If specified we start from checkpoint
    if opt.weights is not None:
        if opt.weights.endswith(".pth") or opt.weights.endswith('.weights'):
            model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
            _, name = os.path.split(opt.weights)
            name, ext = os.path.splitext(name)
            load_epoch = int(name.split("_")[-1]) + 1
        else:
            load_epoch = 0
            print("Pretrained weights must end with .pth or .weights. Using pytorch pretrained weights.")
    else:
        load_epoch = 0

    # Fine-tuning: freezing the network after layer 3 of the backbone
    backbone_blocks = [name for name, child in model.backbone[0].named_children()]
    if opt.freeze is not None and opt.freeze <= 4:
        freeze_blocks = backbone_blocks[:opt.freeze+4]
        print("Freezing backbone up to layer{}".format(opt.freeze))
        for name, child in model.backbone[0].named_children():
            for param_name, params in child.named_parameters():
                if name in freeze_blocks:
                    params.requires_grad = False

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
    model = model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

    images_seen = load_epoch*len(dataset)
    init_epoch = load_epoch
    max_epoch = opt.epochs if opt.epochs is not None else int(max_batches * batch_size / len(dataset))
    processed_batches = load_epoch/batch_size

    loss_logger = DictSaver()
    loss_save_path = os.path.join(backup_path, backup_name + "_loss.csv")
    map_logger = DictSaver()
    map_save_path = os.path.join(backup_path, backup_name + "_map.csv")
    iou_logger = DictSaver()
    iou_save_path = os.path.join(backup_path, backup_name + "_iou.csv")
    avg_loss_logger = DictSaver()
    avg_loss_save_path = os.path.join(backup_path, backup_name + "_avg_loss_epoch.csv")
    # Adapted from https://github.com/pytorch/vision/blob/master/references/detection/engine.py
    for epoch in range(init_epoch, max_epoch):
        avg_loss = 0
        batches_done = epoch*nsamples/batch_size
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
                avg_loss += loss_value

                optimizer.zero_grad()
                losses.backward()

                loss_logger.add_data(loss_dict, batches_done)
                if batch_i % 10 == 0:  # print log every 10 batches
                    # ----------------
                    #   Log progress
                    # ----------------

                    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, max_epoch, batch_i, len(dataloader))
                    log_str += "Learning rate={}\n".format(optimizer.state_dict()['param_groups'][0]['lr'])
                    log_str += "Momentum={}\n".format(optimizer.state_dict()['param_groups'][0]['momentum'])
                    log_str += "\n-- Losses --\n"
                    loss_str = [lk + ": " + str(lv).strip("[]") for lk, lv in loss_dict.items()]
                    log_str += "\n".join(loss_str)
                    log_str += "\n-- Total loss: {}\n".format(losses.item())

                    # Determine approximate time left for epoch
                    epoch_batches_left = len(dataloader) - (batch_i + 1)
                    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
                    log_str += "\n---- ETA {}".format(time_left)

                    print(log_str)

                optimizer.step()
            avg_loss /= len(dataloader)
            print("--Average batch loss: {}\n".format(avg_loss))
            avg_loss_logger.add_data({'avg_loss': [avg_loss]}, epoch)
            avg_loss_logger.save(avg_loss_save_path)

        loss_logger.save(loss_save_path)
        
        if train and epoch % opt.checkpoint_interval == 0:
            save_path = os.path.join(backup_path, backup_name + "_epoch_%d.weights" % epoch)
            torch.save(model.state_dict(), save_path)
            print("Model saved at %s" % save_path)

        if valid:
            if epoch % opt.evaluation_interval == 0:
                print("\n---- Evaluating Model ----\n")
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
                        images = [img.to(device) for img in imgs_eval]
                        try:
                            outputs = model(images)  # List[Dict[Tensor]] with Dict containing 'boxes', 'labels' and 'scores'
                        except RuntimeError as e:
                            print("Runtime Error with image ", imgs_paths)
                            print("Error: {}".format(e))
                        det_boxes_batch, det_labels_batch, det_scores_batch = \
                            evaluation.postprocess_batch_fasterrcnn(outputs, conf_thresh, nms_thresh, num_classes, device)

                        # Store GT for mAP calculation
                        boxes = [t['boxes'].to(device) for t in targets_eval]
                        labels = [t['labels'].to(device) for t in targets_eval]

                        det_boxes.extend(det_boxes_batch)
                        det_labels.extend(det_labels_batch)
                        det_scores.extend(det_scores_batch)
                        true_boxes.extend(boxes)
                        true_labels.extend(labels)

                    # Compute mAP
                    APs, mAP, IoU = evaluation.compute_map(det_boxes, det_labels, det_scores, true_boxes, true_labels,
                                                     num_classes, device, iou_thresh)
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
        # Evaluate the model on the validation set
        model.eval()
        Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        batch_metrics = []
        labels = []
        for eval_i, (_, imgs_eval, targets_eval) in enumerate(tqdm.tqdm(valid_dataloader, desc="Detecting objects")):
            images = list(img.to(device) for img in imgs_eval)
            labels += [t['labels'].tolist() for t in targets_eval]
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
        labels = [item for sublist in labels for item in sublist]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", AP.mean()),
            ("val_f1", f1.mean()),
        ]
        if tf_board:
            logger.list_of_scalars_summary("Evaluation", evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[i, class_names[i], "%.5f" % AP[i]]]
        print(AsciiTable(ap_table).table)
        print("---- mAP {}".format(AP.mean()))
        map_logger.add_data({'mAP': AP.mean()}, batches_done)

    if tf_board:
        logger.close()
    print("Normal ending of the program.")
