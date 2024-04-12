Fine-tuning SAM for openable part segmentation 

Install requirements first - conda create --name <env> --file requirements.txt

Data - rendered data is available at [https://huggingface.co/datasets/diliash/partnetsim-256-fixed-viewpoints](https://huggingface.co/datasets/diliash/partnetsim-256-fixed-viewpoints/tree/master) and https://huggingface.co/datasets/diliash/partnetsim-1024-fixed-viewpoints, it should be just placed into the project root "./". For evaluation you will also need to specify --data_dir and --gt_path, these are availble through our lab network at /localhome/diliash/projects/opmotion/proj-opmotion/data/dataset (PartNet-Mobility dataset) and /localhome/diliash/projects/opmotion/proj-opmotion/minsu3d/visualize/partnetsim/partnetsim-nonoriented-gt (processed PartNet-Mobility) respectively. 

Training notebooks - finetune-pl-detr, fineune-pl-deformable-detr, finetune-pl-sam. Launch them to replicate training results. 

Inference is provided via inference.ipynb, which loads checkpoints from Hugging Face (already public), saves .pt file with inference data. 

Inference data can be loaded and mapped to mesh via sam_to_mesh.py. It will be exported into the format that is read by evaluation code. 

In order to evaluate, use mesh_eval.py for CA, NCA and ARI and mesh_eval_no_base_map.py for P, R and F1.
