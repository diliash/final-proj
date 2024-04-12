import json
import numpy as np 
import argparse
import glob
import os
import json

def compute_bbox_iou(aabb1, aabb2):
    # https://pbr-book.org/3ed-2018/Geometry_and_Transformations/Bounding_Boxes#:~:text=The%20intersection%20of%20two%20bounding,Intersection%20of%20Two%20Bounding%20Boxes.
    (center1, dim1, cat1) = aabb1
    (center2, dim2, cat2) = aabb2
    if int(cat1) != int(cat2):
        return -np.inf

    aabb1_min, aabb1_max = center1 - dim1 / 2, center1 + dim1 / 2
    aabb2_min, aabb2_max = center2 - dim2 / 2, center2 + dim2 / 2
    max_min = np.maximum(aabb1_min, aabb2_min)
    min_max = np.minimum(aabb1_max, aabb2_max)

    intersection_dims = np.maximum(0, min_max - max_min)
    intersection_volume = np.prod(intersection_dims)

    gt_volume = np.prod(aabb1_max - aabb1_min)
    pred_volume = np.prod(aabb2_max - aabb2_min)
    union_volume = gt_volume + pred_volume - intersection_volume

    return intersection_volume / union_volume

def greedy_matching(list1, list2, iou_threshold):
    num_boxes_list1 = len(list1)
    num_boxes_list2 = len(list2)
    max_dim = max(num_boxes_list1, num_boxes_list2)
    distance_matrix = -np.ones((max_dim, max_dim))
    row_indices, col_indices = -np.ones(max_dim, dtype=int), -np.ones(max_dim, dtype=int)

    for i, bbox1 in enumerate(list1):
        current_matching_iou = -np.inf
        current_matching_index = -1
        for j, bbox2 in enumerate(list2):
            iou = compute_bbox_iou(bbox1, bbox2)
            if iou < iou_threshold:
                continue
            if iou > current_matching_iou and j not in col_indices:
                current_matching_index = j
                current_matching_iou = iou
            distance_matrix[i][j] = iou 
        row_indices[i] = i
        col_indices[i] = current_matching_index
    return row_indices, col_indices, distance_matrix

def precision_recall_f1(args):
    tp = 0
    fp = 0
    fn = 0

    per_class_tp = {0: 0, 1: 0, 2: 0}
    per_class_fp = {0: 0, 1: 0, 2: 0}
    per_class_fn = {0: 0, 1: 0, 2: 0}

    for model_id in args.model_ids:
        with open(f"{args.predict_dir}/segmentation_map/{model_id}.json") as f:
            pred_segmentation_map = json.load(f)
        with open(f"{args.gt_path}/gt_segmentation_map/{model_id}.json") as f:
            gt_segmentation_map = json.load(f)

        gt_bboxes = []
        instance_id_idx_map_gt = {}
        for idx, (instance_id, gt_segment) in enumerate(gt_segmentation_map.items()):
            sem_label = gt_segment["semantic"]
            if int(sem_label) != 3:
                instance_id_idx_map_gt[instance_id] = idx
                triangles = np.asarray(gt_segment["triangles"])
                vertices = triangles.reshape(-1, 3)
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)

                center = (bbox_max + bbox_min) / 2
                dim = bbox_max - bbox_min

                gt_bboxes.append((center, dim, sem_label))
        
        pred_bboxes = []
        instance_id_idx_map_pred = {}
        for idx, (pred_instance_id, pred_segment) in enumerate(pred_segmentation_map.items()):
            if "semantic" in pred_segment.keys():
                sem_label = pred_segment["semantic"]
            else:
                # MeshWalker case
                sem_label = pred_instance_id
            if int(sem_label) != 3:
                instance_id_idx_map_pred[pred_instance_id] = idx
                triangles = np.asarray(pred_segment["triangles"])
                vertices = triangles.reshape(-1, 3)
                bbox_min = np.min(vertices, axis=0)
                bbox_max = np.max(vertices, axis=0)

                center = (bbox_max + bbox_min) / 2
                dim = bbox_max - bbox_min

                pred_bboxes.append((center, dim, sem_label))

        matched_gt_indices, matching_pred_indices, distance_matrix = greedy_matching(gt_bboxes, pred_bboxes, args.iou)
        current_tp = np.sum(matching_pred_indices >= 0)
        current_fp = len(pred_bboxes) - current_tp
        current_fn = len(gt_bboxes) - current_tp
        
        tp += current_tp
        fp += current_fp
        fn += current_fn

        for i in range(len(matching_pred_indices)):
            if matching_pred_indices[i] > 0:
                per_class_tp[int(pred_bboxes[matching_pred_indices[i]][2])] += 1
            else:
                if i < len(gt_bboxes):
                    per_class_fn[int(gt_bboxes[i][2])] += 1
                if i < len(pred_bboxes):
                    per_class_fp[int(pred_bboxes[matching_pred_indices[i]][2])] += 1    

        
    macro_precisions = [per_class_tp[c] / (per_class_tp[c] + per_class_fp[c]) if (per_class_tp[c] + per_class_fp[c]) > 0 else 0 for c in per_class_tp]
    macro_recalls = [per_class_tp[c] / (per_class_tp[c] + per_class_fn[c]) if (per_class_tp[c] + per_class_fn[c]) > 0 else 0 for c in per_class_tp]
    macro_f1s = [2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 for precision, recall in zip(macro_precisions, macro_recalls)]
    
    macro_precision = np.mean(macro_precisions)
    macro_recall = np.mean(macro_recalls)
    macro_f1 = np.mean(macro_f1s)
    
    per_class_macro_dict = {c: {"P": macro_precisions[c], "R": macro_recalls[c], "F1": macro_f1s[c]} for c in per_class_tp}

    micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    micro_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {"Micro Prec": micro_precision, "Micro Rec": micro_recall, "Micro F1": micro_f1,
            "Macro Prec": macro_precision, "Macro Rec": macro_recall, "Macro F1": macro_f1,
            "Per Class": per_class_macro_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--predict_dir', type=str,
                        default='./minsu3d/visualize/partnetsim/pred',
                        help='predictions directory')
    parser.add_argument('-d', '--data_dir', type=str,
                       default='./data/dataset')
    parser.add_argument('-g', '--gt_path', type=str,
                       default='./minsu3d/visualize/partnetsim/gt')
    parser.add_argument('-o', '--output_dir', type=str, 
                        default='./mesh_segmentation_results')
    parser.add_argument('--iou', default=0.5, type=float,
                        help="Specify IoU threshold")
    
    args = parser.parse_args()
    args.model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.predict_dir}/segmentation_map/*.json")]
    os.makedirs(f"{args.output_dir}/{args.iou}", exist_ok=True)

    metrics = precision_recall_f1(args)
    print(metrics)

    with open(f"{args.output_dir}/{args.iou}/metrics.json", "w+") as f:
        json.dump(metrics, f)





    