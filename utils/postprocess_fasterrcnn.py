import torch
import numpy as np
from torchvision.ops import boxes as boxes
from utils.utils import bbox_iou
import torchvision.ops.boxes as torch_boxes


def postprocess(outputs, conf_thresh, nms_thresh):
    """
    Compute evalutation metrics.
    :param outputs: from Faster R cnn model, List of Dicts of Tensors, with the Dicts containing:
                    - 'boxes' (FloatTensor[N, 4])
                    - 'labels' (Int64Tensor[N])
                    - scores (Tensor[N])
    :param targets: ground truth data, List of Dicts of Tensors, with the Dicts containing:
                    - 'boxes' (FloatTensor[N, 4])
                    - 'labels' (Int64Tensor[N])
    :return: boxes only, after non maximum suppression
    """
    predictions = []
    for i, output in enumerate(outputs):
        boxes_all = output['boxes']
        scores_all = output['scores']
        labels_all = output['labels']

        # Remove boxes with confidence (score) below confidence threshold
        indices = scores_all > conf_thresh
        boxes = boxes_all[indices]
        
        scores = scores_all[indices]
        labels = labels_all[indices]
        
        keep_indices = torch_boxes.nms(boxes, scores, nms_thresh)  # Int64Tensor
        
        boxes = boxes[keep_indices]
        scores = scores[keep_indices]
        labels = labels[keep_indices]

        pred = torch.zeros((len(boxes), 6))  # Tensor(x1, y1, x2, y2, class score, class prediction)
        pred[:, :4] = boxes
        pred[:, 4] = scores
        pred[:, 5] = labels

        predictions.append(pred)  # List of tensors, one per image in the batch
    return predictions


def batch_statistics(outputs, targets, iou_threshold):
    """
    Compute true positives, predicted scores and predicted labels per sample
    :param outputs: List of Tensors of predictions [x0, y0, x1, y1, confidence, class label]
    :param targets: List of Dicts of Tensors (Dicts keys : 'boxes', 'labels', 'imaeg_id', 'area')
    :param iou_threshold:
    :return:list of True positive, pred_scores and pred_labels for each image in the batch
    """
    batch_metrics = []
    for sample_i in range(len(outputs)):  # outputs[sample_i] is one tensor of several detections, corresponding to one image in the batch
        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4].cpu()
        pred_labels = output[:, -1].cpu()
        # print("\npred_boxes=", pred_boxes)
        #print("pred_scores=", pred_scores)
        #print("pred_labels=", pred_labels)

        true_positives = np.zeros(pred_boxes.shape[0])
        #print("true positives empty=", true_positives)

        annotations = targets[sample_i]['boxes']
        #print("GT boxes=", annotations)
        target_labels = targets[sample_i]['labels'].cpu() if len(annotations)>0 else []
        #print("GT labels=", target_labels)
         
        if len(annotations)>0:
            detected_boxes = []
            target_boxes = annotations

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                #print("\nPrediction ", pred_i)
                #print("pred_box=", pred_box)
                #print("pred_label=", pred_label)
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label.item() not in target_labels:
                    continue
                # print("Input to bbox iou 1 =", pred_box.unsqueeze(0))
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                #print("iou=", iou)
                #print("box_index=", box_index)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    #print("True Positive")
                    detected_boxes += [box_index]
        
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics
