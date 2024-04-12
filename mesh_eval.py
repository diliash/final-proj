import json
import open3d as o3d
import trimesh
import numpy as np 
import argparse
import glob
from tqdm import tqdm
import sklearn
from minsu3d.minsu3d.evaluation.object_detection import evaluate_bbox_acc, get_gt_bbox
import os
import json


def triangle_area_by_coords(coords):
    coords_arr = np.asarray(coords, dtype=float)
    vec1 = coords_arr[1] - coords_arr[0]
    vec2 = coords_arr[2] - coords_arr[0]
    return 0.5 * np.linalg.norm(np.cross(vec1, vec2))

def pred_gt_correspondence(pred_segmentation_map, gt_triangles, n_gt_instances):
    correspondences = {}
    for instance_id, segm_info in pred_segmentation_map.items():
        pred_segm_triangles = np.asarray(segm_info["triangles"])
        gt_instance_votes = np.zeros(n_gt_instances, dtype=int)
        for triangle in pred_segm_triangles:
            tuple_triangle = tuple([tuple(vertex) for vertex in triangle])
            gt_triangle_instance = gt_triangles[str(tuple_triangle)]["instance"]
            gt_instance_votes[gt_triangle_instance] += 1
        correspponding_gt_instance = gt_instance_votes.argmax()
        correspondences[instance_id] = correspponding_gt_instance
    return correspondences

def classification_error(args):
    acc_error = 0.0
    n_scenes = 0
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1
        with open(f"{args.gt_path}/gt_segmentation_map/{scene_id}.json") as file:
            gt_segmentation_map = json.load(file)
        with open(f"{args.predict_dir}/triangles/{scene_id}.json") as file:
            pred_triangles = json.load(file)
        
        scene_error = 0.0
        total_area = 0.0
        # Iterating over GT segmentation just for convenience
        for instance_id, segm_info in gt_segmentation_map.items():
            gt_segm_triangles = segm_info['triangles']
            gt_sem_label = int(segm_info['semantic'])
            
            gt_segm_area = 0.0

            for triangle in gt_segm_triangles:
                tuple_triangle = tuple([tuple(vertex) for vertex in triangle])
                if str(tuple_triangle) in pred_triangles.keys():
                    gt_segm_area += triangle_area_by_coords(tuple_triangle)
                    
            total_area += gt_segm_area
            segm_error = 0.0


            for triangle in gt_segm_triangles:
                tuple_triangle = tuple([tuple(vertex) for vertex in triangle])
                if str(tuple_triangle) in pred_triangles.keys():
                    pred_triangle_info = pred_triangles[str(tuple_triangle)]
                    pred_sem_label = int(pred_triangle_info["semantic"])
                    face_area = triangle_area_by_coords(tuple_triangle)
                    err_val = 0.0
                    if gt_sem_label == pred_sem_label:
                        err_val = 1.0
                    segm_error += err_val * face_area
            scene_error += round(segm_error, 6)

        scene_error /= total_area
        acc_error += scene_error
    acc_error /= n_scenes
    return round(acc_error, 6)

def segment_weighted_error(args):
    acc_error = 0.0
    acc_error_per_class = {}
    n_scenes = 0
    n_scenes_per_class = {}
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1
        with open(f"{args.gt_path}/gt_segmentation_map/{scene_id}.json") as file:
            gt_segmentation_map = json.load(file)
        with open(f"{args.predict_dir}/triangles/{scene_id}.json") as file:
            pred_triangles = json.load(file)

        scene_error = 0.0
        scene_error_per_class = {}

        total_area = 0.0

        #print(scene_id)

        gt_area_per_class = {}

        for _, segm_info in gt_segmentation_map.items():
            gt_segm_triangles = segm_info['triangles']
            gt_sem_label = int(segm_info['semantic'])

            if gt_sem_label not in gt_area_per_class.keys():
                gt_area_per_class[gt_sem_label] = 0.0

            for triangle in gt_segm_triangles:
                tuple_triangle = tuple([tuple(vertex) for vertex in triangle])
                gt_area_per_class[gt_sem_label] += triangle_area_by_coords(tuple_triangle)


        for instance_id, segm_info in gt_segmentation_map.items():
            gt_segm_triangles = segm_info['triangles']
            gt_sem_label = int(segm_info['semantic'])
            semantic_error = 0.0

            for triangle in gt_segm_triangles:
                tuple_triangle = tuple([tuple(vertex) for vertex in triangle])
                if str(tuple_triangle) in pred_triangles.keys():
                    pred_triangle_info = pred_triangles[str(tuple_triangle)]
                    pred_sem_label = int(pred_triangle_info["semantic"])
                    face_area = triangle_area_by_coords(tuple_triangle)
                    err_val = 0.0
                    if gt_sem_label == pred_sem_label:
                        err_val = 1.0
                    scene_error += round(err_val * face_area / gt_area_per_class[gt_sem_label], 6)
                    semantic_error += round(err_val * face_area / gt_area_per_class[gt_sem_label], 6)
            if segm_info['semantic'] in scene_error_per_class.keys():
                scene_error_per_class[segm_info['semantic']] += round(semantic_error, 6)
            else:
                scene_error_per_class[segm_info['semantic']] = round(semantic_error, 6)
        scene_error /= np.unique(list(gt_area_per_class.keys())).shape[0]
        for key in scene_error_per_class.keys():
            if key in n_scenes_per_class.keys():
                n_scenes_per_class[key] += 1
            else:
                n_scenes_per_class[key] = 1
    
            if key in acc_error_per_class.keys():
                acc_error_per_class[key] += scene_error_per_class[key]
            else: 
                acc_error_per_class[key] = scene_error_per_class[key]
            
        #print(scene_error)
        acc_error += scene_error
    acc_error /= n_scenes
    print("Segment-Weighted Error Per Class:")

    for key in acc_error_per_class.keys():
        acc_error_per_class[key] /= n_scenes_per_class[key]
        print(f"\t{args.id_class_map[key]}: {round(acc_error_per_class[key], 6)}")
    
    return round(acc_error, 6), acc_error_per_class

def rand_index(args):
    n_scenes = 0
    acc_ari = 0.0
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1
        with open(f"{args.gt_path}/gt_segmentation_map/{scene_id}.json") as file:
            gt_segmentation_map = json.load(file)
        with open(f"{args.gt_path}/gt_triangles_map/{scene_id}.json") as file:
            gt_triangles = json.load(file)
        with open(f"{args.predict_dir}/triangles/{scene_id}.json") as file:
            pred_triangles = json.load(file)

        gt_instances_list = [int(idx) for idx in gt_segmentation_map.keys()]
        
        pred_labels = []
        gt_labels = []
        
        for str_triangle, pred_triangle_info in pred_triangles.items():
            gt_triangle_info = gt_triangles[str_triangle]
            gt_labels.append(int(gt_triangle_info['instance']))
            #pred_labels.append(instance_correspondences[str(pred_triangle_info["instance"])])
            pred_labels.append(int(pred_triangle_info["instance"]))

        acc_ari += sklearn.metrics.adjusted_rand_score(gt_labels, pred_labels)

    acc_ari /= n_scenes
    return acc_ari

def AP_bbox(args):
    n_scenes = 0
    acc_ap = 0.0
    all_pred_instances = []
    all_gt_bboxes = []
    for scene_id in tqdm(args.model_ids):
        n_scenes += 1

        scene_preds = []
        scene_bboxes = []

        with open(f"{args.gt_path}/gt_segmentation_map/{scene_id}.json") as file:
            gt_segmentation_map = json.load(file)
        with open(f"{args.gt_path}/gt_triangles_map/{scene_id}.json") as file:
            gt_triangles = json.load(file)
        with open(f"{args.predict_dir}/triangles/{scene_id}.json") as file:
            pred_triangles = json.load(file)
        with open(f"{args.predict_dir}/segmentation_map/{scene_id}.json") as file:
            pred_segmentation_map = json.load(file)
        
        for segm_id, segm_info in pred_segmentation_map.items():
            pred = {"scan_id": scene_id, "label_id": segm_info["semantic"],
                    "conf": 1.0}
            
            pred_triangles = np.asarray(segm_info["triangles"])
            pred_xyz = []
            for triangle in pred_triangles:
                for vertex in triangle:
                    pred_xyz.append(vertex)
            pred_xyz = np.asarray(pred_xyz)
            pred["pred_bbox"] = np.concatenate((pred_xyz.min(0), pred_xyz.max(0)))
            scene_preds.append(pred)

        for segm_id, segm_info in gt_segmentation_map.items():
            gt_triangles = np.asarray(segm_info["triangles"])
            gt_xyz = []
            for triangle in gt_triangles:
                for vertex in triangle:
                    gt_xyz.append(vertex)
            gt_xyz = np.asarray(gt_xyz)
            scene_bboxes.append((int(segm_info["semantic"]), np.concatenate((gt_xyz.min(0), gt_xyz.max(0)))))

        all_pred_instances.append(scene_preds)
        all_gt_bboxes.append(scene_bboxes)
    #print(all_pred_instances)
    #print(all_gt_bboxes)
    obj_detect_eval_result = evaluate_bbox_acc(all_pred_instances, all_gt_bboxes, ["drawer", "door", "lid", "base"],
                                               [], print_result=True)
    return
        

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
    
    args = parser.parse_args()
    args.model_ids = [path.split('/')[-1].split('.')[0] for path in glob.glob(f"{args.predict_dir}/segmentation_map/*.json")]
    args.id_class_map = {0: "drawer", 1: "door", 2: "lid", 3: "base"}
    os.makedirs(args.output_dir, exist_ok=True)

    ce = classification_error(args)
    print(f"Average Classification error: {ce}")
    swe, swe_cls = segment_weighted_error(args)
    print(f"Average Segment-weighted error: {swe}")
    ri = rand_index(args)
    print(f"Average Adjusted Rand Index: {ri}")

    eval_dict = {"Classification Accuracy": {"average": ce}, "Normalized Classification Accuracy": {"average": swe, "class": swe_cls}, "ARI": ri}
    with open(f"{args.output_dir}/eval_dict.json", "w+") as f:
        json.dump(eval_dict, f)
    #print(AP_bbox(args))
    






    