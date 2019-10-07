# Modified from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection
import torch
import torchvision.ops


def postprocess_batch_fasterrcnn(outputs, conf_thresh, nms_thresh, n_classes, device):
    """
    Process the batch results
    :param outputs:
    :param conf_thresh:
    :param nms_thresh:
    :param n_classes:
    :return: all_images_boxes, all_images_labels, all_images_scores
    """
    # Separate boxes and labels for the batch
    det_boxes_batch = [d['boxes'] for d in outputs]
    #det_boxes_batch = torch.tensor(det_boxes_batch)  # Tensor (N, n boxes per image, 4)
    
    det_scores_batch = [d['scores'] for d in outputs]
    # det_scores_batch = torch.tensor(det_scores_batch)  # Tensor (N, n scores per image, 1)

    det_labels_batch = [d['labels'] for d in outputs]
    # det_labels_batch = torch.tensor(det_labels_batch)  # Tensor (N, labels of detections per image, 1)

    batch_size = len(det_boxes_batch)

    # Lists to store final outputs
    all_images_boxes = list()
    all_images_labels = list()
    all_images_scores = list()

    for i in range(batch_size):  # For each image in the batch
        image_boxes = list()
        image_labels = list()
        image_scores = list()
        
        for c in range(1, n_classes):
            # Keep boxes and scores when score is above threshold for the particular class
            score_above_min = (det_scores_batch[i] > conf_thresh) * (det_labels_batch[i] == c)
            # print("score above min=", score_above_min)
            n_above_min_score = score_above_min.sum().item()
            if n_above_min_score == 0:
                continue
            
            class_scores = det_scores_batch[i][score_above_min]
            class_boxes = det_boxes_batch[i][score_above_min]
            class_labels = det_labels_batch[i][score_above_min]

            # Sort boxes and scores by score
            class_scores, sort_ind = class_scores.sort(dim=0, descending=True)
            class_boxes = class_boxes[sort_ind]
            class_labels = class_labels[sort_ind]

            # Compute nms
            keep_indices = torchvision.ops.nms(class_boxes, class_scores, nms_thresh)
            image_boxes.append(class_boxes[keep_indices])
            image_labels.append(class_labels[keep_indices])
            image_scores.append(class_scores[keep_indices])
        
        # If no object in any class found, store a placeholder for 'background'
        if len(image_boxes) == 0:
           image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
           image_labels.append(torch.LongTensor([0]).to(device))
           image_scores.append(torch.FloatTensor([0.]).to(device))
                 
        # Concatenate into single tensors
        image_boxes = torch.cat(image_boxes, dim=0)  # (n_objects, 4)
        image_labels = torch.cat(image_labels, dim=0)  # (n_objects)
        image_scores = torch.cat(image_scores, dim=0)  #

        all_images_boxes.append(image_boxes)
        all_images_labels.append(image_labels)
        all_images_scores.append(image_scores)
         
    return all_images_boxes, all_images_labels, all_images_scores


def compute_map(det_boxes, det_labels, det_scores, true_boxes, true_labels, n_classes, device, iou_thresh):
    """

    :param det_boxes: list of Tensors (n_objects, 4)
    :param det_labels: list of Tensors (n_objects)
    :param det_scores: list of Tensors (n_objects)
    :param true_boxes:
    :param true_labels:
    :return: list of AP for all classes, mAP
    """
    assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)  # number of images
    true_images = list()
    for i in range(len(true_labels)):
        true_images.extend([i] * true_labels[i].size(0))
    true_images = torch.LongTensor(true_images).to(device)
    # true_images has one place for each true object, with value the index of each image (0 for first image, 1 for second image etc...)

    # True detections are separated by batch
    true_boxes = torch.cat(true_boxes, dim=0)  # (n_objects, 4)
    true_labels = torch.cat(true_labels, dim=0)  # (n_objects)

    assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

    # Store all detections in a single continuous tensor while keeping track of the image it is from
    det_images = list()
    for i in range(len(det_labels)):
        det_images.extend([i] * det_labels[i].size(0))
    det_images = torch.LongTensor(det_images).to(device)
    # det_images has one place for each detected object, with value the index of each image

    # Detections are separated by batch
    det_boxes = torch.cat(det_boxes, dim=0)  # (n_detections, 4)
    det_labels = torch.cat(det_labels, dim=0)  # (n_detections)
    det_scores = torch.cat(det_scores, dim=0)  # (n_detections)

    # Compute AP for each class
    print("n classes = ", n_classes)
    average_precisions = torch.zeros(n_classes-1, dtype=torch.float)  # (n_classes)
    print("empty APs=", average_precisions)
    for c in range(1, n_classes):
        # print("true labels=", true_labels)
        # print("true images=", true_images)
        # Extract only objects with this class
        true_class_images = true_images[true_labels == c]  # (n_class_objects), images with true objects from this class
        # print("true class images=", true_class_images)        
        true_class_boxes = true_boxes[true_labels == c]  # (n_class_objects, 4), boxes of the true objects from this class
        n_class_objects = true_class_boxes.size(0)

        # Keep track of which true objects from this class have been detected
        true_class_boxes_detected = torch.zeros((true_class_boxes.size(0)), dtype=torch.uint8).to(device)  # (n_class_objects)
        
        # Extract only detections from this class
        det_class_images = det_images[det_labels == c]
        det_class_boxes = det_boxes[det_labels == c]
        det_class_scores = det_scores[det_labels == c]
        n_class_detections = det_class_boxes.size(0)
        if n_class_detections == 0:
            # No detection from this class
            continue

        # Sort detections in decreasing order of score
        det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)  # (n_class_detections)
        det_class_images = det_class_images[sort_ind]  # (n_class_detections)
        det_class_boxes = det_class_boxes[sort_ind]  # (n_class_detections, 4)

        # Check if a detection is a true or false positive
        true_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)
        false_positives = torch.zeros((n_class_detections), dtype=torch.float).to(device)
        for d in range(n_class_detections):
            this_detection_box = det_class_boxes[d].unsqueeze(0)  # (1, 4)
            this_image = det_class_images[d]  # (), scalar

            # Find objects in the same imagewith this class, and whether they have been detected before
            object_boxes = true_class_boxes[true_class_images == this_image]  # (n_class_objects_in_img)
            # if no object of this class in this image : false positive
            if object_boxes.size(0) == 0:
                false_positives[d] = 1
                continue

            # Find maximum overlap of this detection with objects in this image of this class
            overlaps = find_jaccard_overlap(this_detection_box, object_boxes)  # (1, n_class_objects_in_img
            max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)  # (), () - scalars

            # ind is the index of the object in the image-level tensors object_boxes
            # Compute index in the original class-level tensors:
            original_ind = torch.LongTensor(range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

            # If the maximum overlap is greater than the threshold, it is a match
            if max_overlap.item() > iou_thresh:
                # If this object is not already detected, it is a true positive
                if true_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    true_class_boxes_detected[original_ind] = 1
                # Otherwise, it is a false positive
                else:
                    false_positives[d] = 1
            # Otherwise, it is a false positive
            else:
                false_positives[d] = 1

        # Compute cumulative precision and recall at each detection, in the order of decreasing score
        cumul_true_positives = torch.cumsum(true_positives, dim=0)  # (n_class_detections)
        cumul_false_positives = torch.cumsum(false_positives, dim=0)  # (n_class_detections)
        cumul_precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + 1e-10)  # (n_class_detections)
        cumul_recall = cumul_true_positives / n_class_objects  # (n_class_detections)

        # Find mean of the max of precisions corresponding to recalls above threshold t
        recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()  # (11)
        precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float).to(device)  # (11)
        for i, t in enumerate(recall_thresholds):
            recalls_above_t = cumul_recall >= t
            if recalls_above_t.any():
                precisions[i] = cumul_precision[recalls_above_t].max()
            else:
                precisions[i] = 0
        average_precisions[c-1] = precisions.mean()
        print("average_precisions=", average_precisions)

    # Compute mAP
    mean_average_precision = average_precisions.mean().item()

    return average_precisions, mean_average_precision


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)
