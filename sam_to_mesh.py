import json
import pycocotools
import numpy as np
import trimesh
import argparse
import open3d as o3d
from PIL import Image
import os
import copy
from glob import glob
import torch
import torchvision

PARTNETSIM_COLOR_MAP_RGBA = {
    1: (0, 107, 164, 255),
    2: (255, 128, 14, 255),
    3: (200, 82, 0, 255),
    4: (171, 171, 171, 255),
}

IOU_THRESHOLD = 0.8

def triangle_area(coordinates):
    p1, p2, p3 = coordinates
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p1)
    area_vector = np.cross(v1, v2)
    return np.linalg.norm(area_vector) / 2.0

def compute_triangle_areas(triangle_dict):
    areas = np.array([triangle_area(triangle_dict[i]) for i in triangle_dict])
    return areas

def compute_iou(mask1, mask2, triangle_dict):
    mask1 = mask1.astype(np.bool_)
    mask2 = mask2.astype(np.bool_)
    
    triangle_areas = compute_triangle_areas(triangle_dict)
    
    intersection = np.logical_and(mask1, mask2).astype(np.float32) * triangle_areas
    union = np.logical_or(mask1, mask2).astype(np.float32) * triangle_areas
    
    iou = intersection.sum() / union.sum()
    return iou

def sort_dicts_by_field(dicts, field):
    return sorted(dicts, key=lambda x: x[field], reverse=True)

def color_to_index(color):
    return color[0] * 256 * 256 + color[1] * 256 + color[2]

class CocoLoader(torchvision.datasets.CocoDetection):
    def __init__(self, coco_folder, train=True):
        ann_file = os.path.join(coco_folder, "coco_annotation", "MotionNet_train.json" if train else "MotionNet_valid.json")
        super(CocoLoader, self).__init__(os.path.join(coco_folder, "train/origin" if train else "valid/origin"), ann_file)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        anns = self.coco.loadImgs(image_id)[0]
        image_id = anns.pop("id")
        anns["image_id"] = image_id
        return anns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--pred_path', type=str, default='./aggregated_masks.pt', 
                        help='Path to predictions ')
    parser.add_argument('-d', '--dataset', type=str, default='./obj-partnetsim-textured-notransform', 
                        help='specify path to dataset')
    parser.add_argument('-s', '--stk', type=str, default='./partnetsim-1024-fixed-viewpoints', 
                        help='Path to original STK data folder')
    parser.add_argument('-e', '--export_path', type=str, default='./sam_exported_predictions', 
                        help='Path to original STK data folder')
    
    args = parser.parse_args()
    os.makedirs(f"{args.export_path}/triangles", exist_ok=True)
    os.makedirs(f"{args.export_path}/segmentation_map", exist_ok=True)
    os.makedirs(f"{args.export_path}/pcd", exist_ok=True)
    os.makedirs(f"{args.export_path}/segmentation_mask", exist_ok=True)
    os.makedirs(f"{args.export_path}/motion", exist_ok=True)

    model_ids = np.sort([path.split("/")[-1].split(".")[1] for path in glob(f"./{args.stk}/triangle_maps/*")])

    sam_preds = torch.load(args.pred_path)
    coco_dataset = CocoLoader(args.stk + "/coco", train=False)

    image_id_filename_map = {}
    image_id = 0
    for model_id in model_ids:
        for i in range(3):
            filename = f"{model_id}.png-{i}.png"
            image_id_filename_map[str(image_id)] = filename
            image_id += 1

    image_id_filename_map = {}
    image_id_extrinsic_map = {}
    for d in coco_dataset:
        filename = d["file_name"].split("/")[-1].split(".")[0]
        image_id = d["image_id"]
        image_id_extrinsic_map[str(image_id)] = d["camera"]["extrinsic"]["matrix"]
        image_id_filename_map[str(image_id)] = filename

    updated_preds = {}
    new_id = 0
    for idx, prediction in sam_preds.items():
        image_info = prediction.pop("image_info")[0]
        image_info.pop("label")
        scores = prediction.pop("scores")[0]
        masks = prediction.pop("masks")[0]
        labels = prediction.pop("lables")[0]
        for i, mask in enumerate(masks):
            updated_preds[new_id] = {"score": scores[i], "mask": mask.squeeze().cpu().numpy(), "label": labels[i]}
            updated_preds[new_id].update(image_info)
            image_id = updated_preds[new_id].pop("id")
            updated_preds[new_id]["image_id"] = image_id
            new_id += 1
    model_id_aggregated_predictions = {}
    prediction_id_image_id_map = {}
    for idx, prediction in updated_preds.items():
        prediction["id"] = str(idx)
        prediction_id_image_id_map[str(idx)] = prediction["image_id"]
        model_id = image_id_filename_map[str(prediction["image_id"])].split("-")[0].split(".")[0]
        if model_id in model_id_aggregated_predictions.keys():
            model_id_aggregated_predictions[model_id].append(prediction)
        else:
            model_id_aggregated_predictions[model_id] = [prediction]
    
    for model_id, model_predictions in model_id_aggregated_predictions.items():
        print(model_id)
        os.makedirs(f"{args.export_path}/obj/{model_id}")
        os.makedirs(f"{args.export_path}/png/{model_id}")
        #TODO parallelize this part
        image_id_cache = {}
        with open(f"{args.stk}/triangle_maps/partnetsim.{model_id}.json", "r") as f:
            shape_triangles_map = json.load(f)
        triangle_index_instance_map = -np.ones(len(shape_triangles_map), dtype=np.int_)
        model_predictions = sort_dicts_by_field(model_predictions, "score")
        for prediction in model_predictions:
            image_id = str(prediction["image_id"])
            motion = {"gtextrinsic": image_id_extrinsic_map[image_id],
                      "image_id": image_id}
            prediction_id = prediction["id"]
            current_score = prediction["score"]
            label = prediction["label"]
            if image_id not in image_id_cache.keys():
                im = Image.open(f"{args.stk}/render_2dmotion/{model_id}/triindex/{image_id_filename_map[image_id]}_triindex.png")
                visible_faces = -np.ones([1024, 1024], dtype=np.intc)
                for i in range(1024):
                    for j in range(1024):
                        pixel = im.getpixel((i, j))
                        if pixel[3] != 0:
                            visible_faces[j][i] = color_to_index(pixel)
                mask_instance_map_data = {}
                mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
                mask_instnace_map = -np.ones([1024, 1024], dtype=np.int_)
                image_id_cache[image_id] = [mask_instance_map_data, visible_faces]
            else:
                mask_instance_map_data, visible_faces = image_id_cache[image_id]
                mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
                image_id_cache[image_id] = [mask_instance_map_data, visible_faces]
            mask_decoded = prediction["mask"]
            """from matplotlib import pyplot as plt
            plt.figure(figsize=(6, 6))
            colored_mask = np.zeros((*mask_decoded.shape, 4))
            colored_mask[visible_faces>0] = np.asarray([1, 1, 1, 1])
            colored_mask[mask_decoded > 0] = np.asarray(PARTNETSIM_COLOR_MAP_RGBA[label]) / 255
            print(current_score)
            plt.imshow(colored_mask)
            plt.axis('off')
            plt.show()"""


            faces_mask = np.zeros(len(shape_triangles_map), dtype=np.bool_)
            faces_mask[visible_faces[mask_decoded]] = True
            #TODO make sure cases where mask goes over the edge of object are handled
            
            currently_assigned = triangle_index_instance_map[visible_faces[np.logical_and(mask_decoded, visible_faces != -1)]]
            occupied = np.where(currently_assigned >= 0)[0]

            if len(occupied):
                overlapping_instances = np.unique(currently_assigned[occupied])
                ious = []
                for overlapping_instance in overlapping_instances:
                    overlapping_instance_mask = triangle_index_instance_map == overlapping_instance
                    ious.append({"id": str(overlapping_instance), "iou": compute_iou(faces_mask, overlapping_instance_mask, shape_triangles_map)})
                ious = sort_dicts_by_field(ious, "iou")
                
                recompute_iou = False
                for ious_dict in ious:
                    overlapping_instance, iou = ious_dict.values()
                    if recompute_iou:
                        overlapping_instance_mask = triangle_index_instance_map == int(overlapping_instance)
                        iou = compute_iou(faces_mask, overlapping_instance_mask, shape_triangles_map)
                    overlapping_mask_instance_map_data, overlapping_visible_faces = image_id_cache[str(prediction_id_image_id_map[overlapping_instance])]
                    overlapping_instance_score = overlapping_mask_instance_map_data[str(overlapping_instance)]["score"]
                    overlapping_instance_mask = triangle_index_instance_map == int(overlapping_instance)
                    #if iou >= IOU_THRESHOLD and label == overlapping_mask_instance_map_data[overlapping_instance]["label"]:
                    if iou >= IOU_THRESHOLD:
                        if overlapping_instance_score > current_score:
                            """print(f"Current prediction id: {prediction_id}, image_id: {image_id}")
                            print(f"Cache: {image_id_cache}")"""

                            mask_instance_map_data, visible_faces = image_id_cache[image_id]
                            mask_instance_map_data.pop(prediction_id)
                            image_id_cache[image_id] = [mask_instance_map_data, visible_faces]

                            image_id = str(prediction_id_image_id_map[overlapping_instance])
                            prediction_id_image_id_map[prediction_id] = image_id
                            prediction_id = overlapping_instance

                            """print(f"Instance moved to image_id {image_id}, new prediction id {prediction_id}")
                            print(f"Cache: {image_id_cache}")
                            print()
                            print()"""
                            
                            faces_mask = np.logical_or(faces_mask, overlapping_instance_mask)
                            current_score = overlapping_mask_instance_map_data[str(overlapping_instance)]["score"]
                        else:
                            faces_mask = np.logical_or(faces_mask, overlapping_instance_mask)
                            overlapping_mask_instance_map_data.pop(overlapping_instance)
                            image_id_cache[str(prediction_id_image_id_map[overlapping_instance])] = [overlapping_mask_instance_map_data, overlapping_visible_faces]
                            prediction_id_image_id_map[overlapping_instance] = image_id
                    else:
                        overlapping_mask = np.logical_and(overlapping_instance_mask, faces_mask)
                        if overlapping_instance_score > current_score:
                            faces_mask = np.logical_xor(faces_mask, overlapping_mask)
                        else:
                            faces_mask = np.logical_or(faces_mask, overlapping_mask)

            triangle_index_instance_map[faces_mask] = prediction_id
            mask_instance_map_data, visible_faces = image_id_cache[image_id]
            mask_instance_map_data[prediction_id] = {"score": current_score, "label": label, "motion": motion}
            image_id_cache[image_id] = [mask_instance_map_data, visible_faces]

        
        temp_mesh = trimesh.load(f"{args.dataset}/val/{model_id}/{model_id}.obj", process=False, maintain_order=True, force="mesh")

        flattened_triangles = np.transpose(np.array([triangle.flatten() for triangle in temp_mesh.triangles]))
        kdtree = o3d.geometry.KDTreeFlann(flattened_triangles)

        stk_to_trimesh_triangle_index_map = {}
        trimesh_to_stk_triangle_index_map = {}
        
        for stk_triangle_index, stk_triangle_vertices in shape_triangles_map.items():
            query_triangle = np.array(stk_triangle_vertices).flatten()
            _, idx, _ = kdtree.search_knn_vector_xd(query_triangle, 1)
            corresponding_index = idx[0]
            stk_to_trimesh_triangle_index_map[stk_triangle_index] = corresponding_index
            trimesh_to_stk_triangle_index_map[corresponding_index] = stk_triangle_index
        
        face_colors = np.asarray([PARTNETSIM_COLOR_MAP_RGBA[4]] * len(temp_mesh.triangles), dtype=np.uint8)

        triangles_map = {}
        segmentation_map = {}
        segmentation_mask = -np.ones(len(temp_mesh.triangles))
        for instance_id, prediction_id in enumerate(np.unique(triangle_index_instance_map)):
            stk_indexes = np.where(triangle_index_instance_map == prediction_id)[0]
            trimesh_indexes = np.asarray([stk_to_trimesh_triangle_index_map[str(stk_id)] for stk_id in stk_indexes])
            segmentation_map[str(instance_id)] = {"triangles": [], "semantic": None, "geometries": []}
            if prediction_id == -1:
                face_colors[trimesh_indexes] = np.asarray(PARTNETSIM_COLOR_MAP_RGBA[4])
                segmentation_map[str(instance_id)]["semantic"] = str(3)
                segmentation_mask[trimesh_indexes] = instance_id
                label = 4
                motion = None
                for idx in trimesh_indexes:
                    triangles_data = str(tuple([tuple(np.round(triangle, 6).tolist()) for triangle in temp_mesh.triangles[idx].tolist()]))
                    triangles_map[triangles_data] = {"semantic": str(3), "instance": str(instance_id)}
                    segmentation_map[str(instance_id)]["triangles"].append(np.round(temp_mesh.triangles[idx], 6).tolist())
            else:
                mask_instance_map_data, _ = image_id_cache[str(prediction_id_image_id_map[str(prediction_id)])]
                label = mask_instance_map_data[str(prediction_id)]["label"]
                motion = mask_instance_map_data[str(prediction_id)]["motion"]
                face_colors[trimesh_indexes] = np.asarray(PARTNETSIM_COLOR_MAP_RGBA[label])
                segmentation_map[str(instance_id)]["semantic"] = str(label - 1)
                segmentation_mask[trimesh_indexes] = instance_id
                for idx in trimesh_indexes:
                    triangles_data = str(tuple([tuple(np.round(triangle, 6).tolist()) for triangle in temp_mesh.triangles[idx].tolist()]))
                    triangles_map[triangles_data] = {"semantic": str(label - 1), "instance": str(instance_id)}
                    segmentation_map[str(instance_id)]["triangles"].append(np.round(temp_mesh.triangles[idx], 6).tolist())
            sub_mesh = trimesh.Trimesh(vertices=temp_mesh.vertices, faces=temp_mesh.faces[trimesh_indexes])
            sub_mesh.remove_unreferenced_vertices()
            result = trimesh.sample.sample_surface(sub_mesh, 100000, sample_color=False)
            pcd_points = result[0]
            colors = np.asarray([np.asarray(PARTNETSIM_COLOR_MAP_RGBA[label])[:3] / 255] * 100000)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            downpcd = pcd.voxel_down_sample(voxel_size=0.005) 
            with open(f"./{args.export_path}/pcd/{model_id}-{instance_id}.npz", "wb+") as outfile: 
                np.savez(outfile, points=np.array(downpcd.points), colors=np.array(downpcd.colors), instance=np.asarray(instance_id), semantic=np.asarray(int(label - 1)))
            if motion:
                with open(f"{args.export_path}/motion/{model_id}-{instance_id}.json", "w+") as f:
                    json.dump(motion, f)
        semantic_visuals = trimesh.visual.ColorVisuals(face_colors=face_colors)
        temp_mesh.visual = semantic_visuals

        #temp_mesh.show()

        with open(f"{args.export_path}/triangles/{model_id}.json", "w+") as f:
            json.dump(triangles_map, f)

        with open(f"{args.export_path}/segmentation_map/{model_id}.json", "w+") as f:
            json.dump(segmentation_map, f)
        
        with open(f"{args.export_path}/segmentation_mask/{model_id}.npy", "wb+") as outfile: 
            np.save(outfile, np.asarray(segmentation_mask))

        filename = f"{args.export_path}/obj/{model_id}/{model_id}.obj"


        # NOTE: saved meshes are unreliable - trimesh dosn't export *all* colors properly here
        # Adapted from https://github.com/mikedh/trimesh/issues/729#issuecomment-593656942

        obj = trimesh.exchange.obj.export_obj(temp_mesh, include_color=True, include_texture=False)

        # where is the OBJ file going to be saved                
        obj_path = filename
        with open(obj_path, 'w') as f:
            f.write(obj)

        tranform_mesh_1 = copy.deepcopy(temp_mesh)
        angle1 = 0.85 * np.pi / 3
        axis1 = [0, 1, 0]
        transform1 = trimesh.transformations.rotation_matrix(angle1, axis1)
        angle2 = np.pi / 6
        axis2 = [1, 0, 0]
        transform2 = trimesh.transformations.rotation_matrix(angle2, axis2)
        scene = trimesh.Scene([tranform_mesh_1]).apply_transform(transform1).apply_transform(transform2)
        #scene = trimesh.Scene([temp_mesh]).apply_transform(transform1)
        #png = scene.save_image(resolution=[1024, 512]) 
        #with open(f"{args.export_path}/png/{model_id}/{model_id}-1.png", "wb+") as f:
        #    f.write(png)

        tranform_mesh_2 = copy.deepcopy(temp_mesh)
        angle1 = 0.85 * np.pi / 3 + np.pi
        axis1 = [0, 1, 0]
        transform1 = trimesh.transformations.rotation_matrix(angle1, axis1)
        angle2 = -np.pi / 6
        axis2 = [1, 0, 0]
        transform2 = trimesh.transformations.rotation_matrix(angle2, axis2)
        scene = trimesh.Scene([tranform_mesh_2]).apply_transform(transform1).apply_transform(transform2)
        #scene = trimesh.Scene([temp_mesh]).apply_transform(transform1)
        #png = scene.save_image(resolution=[1024, 512]) 
        #with open(f"{args.export_path}/png/{model_id}/{model_id}-2.png", "wb+") as f:
        #    f.write(png)


        
            
            

                
        
    
        




