import argparse
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
import ast
import json
import shutil

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator


models = [
    "faster_rcnn_R_50_FPN_3x",
    "faster_rcnn_R_101_FPN_3x",
    "faster_rcnn_X_101_32x8d_FPN_3x",
    "retinanet_R_50_FPN_3x",
    "retinanet_R_101_FPN_3x",
]


def get_args_parser():
    parser = argparse.ArgumentParser('Set up Detectron2', add_help=False)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--model_dir', default=None, type=str)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train_annot_fp', default="./annot_train.csv", type=str)
    parser.add_argument('--val_annot_fp', default="./annot_val.csv", type=str)
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--ims_per_batch', default=4, type=int)
    parser.add_argument('--warmup_iters', default=1000, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--lr_decay_steps', default=[100000,], type=int, nargs='*')
    return parser


def get_detectron_dicts(annot_path):
    """
    Create a Detectron2's standard dataset dicts from an annotation file.
    
    Args:
        annot_path (str): path to the annotation file.
    Return:
        dataset_dicts (list[dict]): List of annotation dictionaries for Detectron2.
    """
    print("Loading annotation from ", annot_path)
    # Load annotation DataFrame
    annot_df = pd.read_csv(annot_path)
    
    # Get all images in `annot_df`
    img_ids = annot_df["image_id"].unique().tolist()
    
    # Convert `bbox` column from string to list
    annot_df['bbox'] = annot_df['bbox'].apply(ast.literal_eval)
    
    dataset_dicts = []
    for img_id in tqdm(img_ids):
        file_name = f'train/{img_id}.jpg'
            
        record = {}
        record['file_name'] = file_name
        record['image_id'] = img_id
        record['height'] = 1024
        record['width'] = 1024
        
        # Extract bboxes from annotation file
        bboxes = annot_df[annot_df['image_id'] == img_id]['bbox']
        
        annots = []
        for bbox in bboxes:
            annot = {
                "bbox": bbox,
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": 0,
            }
            annots.append(annot)

        record["annotations"] = annots
        dataset_dicts.append(record)
    return dataset_dicts


def main(args):
    # Register datasets
    print("Registering wheat_detection_train")
    DatasetCatalog.register("wheat_detection_train", lambda path=args.train_annot_fp: get_detectron_dicts(path))
    MetadataCatalog.get("wheat_detection_train").set(thing_classes=["Wheat"])

    print("Registering wheat_detection_val")
    DatasetCatalog.register("wheat_detection_val", lambda path=args.val_annot_fp: get_detectron_dicts(path))
    MetadataCatalog.get("wheat_detection_val").set(thing_classes=["Wheat"])

    # Set up configurations
    cfg = get_cfg()
    if not args.model_dir:
        cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{args.model}.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{args.model}.yaml")
        cfg.DATASETS.TRAIN = ("wheat_detection_train",)
        cfg.DATASETS.TEST = ("wheat_detection_val",)

        cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
        cfg.SOLVER.BASE_LR = args.lr
        cfg.SOLVER.MAX_ITER = args.max_iter
        cfg.SOLVER.WARMUP_ITERS = args.warmup_iters
        cfg.SOLVER.GAMMA = args.gamma
        cfg.SOLVER.STEPS = args.lr_decay_steps

        cfg.DATALOADER.NUM_WORKERS = 6
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.MODEL.RETINANET.NUM_CLASSES = 1
        cfg.OUTPUT_DIR = f"{args.model}__iter-{args.max_iter}__lr-{args.lr}"
        if os.path.exists(cfg.OUTPUT_DIR): shutil.rmtree(cfg.OUTPUT_DIR)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        
        # Save config
        with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
            f.write(cfg.dump())
    else:
        print("Loading model from ", args.model_dir)
        cfg.merge_from_file(os.path.join(args.model_dir, "config.yaml"))
        cfg.MODEL.WEIGHTS = os.path.join(args.model_dir, "model_final.pth")
        cfg.OUTPUT_DIR = args.model_dir

    # Train
    setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "terminal_output.log"))
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    if args.train:
        trainer.train()

    # Evaluate
    if args.eval:
        evaluator = COCOEvaluator("wheat_detection_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
        eval_results = trainer.test(cfg=cfg, model=trainer.model, evaluators=evaluator)
        with open(os.path.join(cfg.OUTPUT_DIR, "eval_results.json"), "w") as f:
            json.dump(eval_results, f)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
