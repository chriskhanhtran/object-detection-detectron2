# Model list
models = [
    "faster_rcnn_R_50_FPN_3x",
    "faster_rcnn_R_101_FPN_3x",
    "faster_rcnn_X_101_32x8d_FPN_3x", # best
    "retinanet_R_50_FPN_3x",
    "retinanet_R_101_FPN_3x",
]

# Class: 'Musical instrument'
target_classes = [
    'Accordion',
    'Cello',
    'Drum',
    'French horn',
    'Guitar',
    'Musical keyboard',
    'Piano',
    'Saxophone',
    'Trombone',
    'Trumpet',
    'Violin'
]

# Arguments
MODEL = models[4]
IMS_PER_BATCH = 4
MAX_ITER = 100000
BASE_LR = 5e-4
WARMUP_ITERS = 1000
GAMMA = 0.5
STEPS = (10000, 30000, 50000, 70000, 90000,)
debug = False

experiment = f"{MODEL}__iter-{MAX_ITER}__lr-{BASE_LR}__warmup-{WARMUP_ITERS}"

########################################################################################################################
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from tqdm import tqdm
import shutil
import os
import glob
import re
import cv2
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode


# Format annotations
def bbox_rel_to_abs(bbox, height, width):
    """
    Converts bounding box dimensions from relative to absolute pixel values (Detectron2 style).
    See: https://detectron2.readthedocs.io/modules/structures.html#detectron2.structures.BoxMode
    
    Args:
        bbox (array): relative bounding box in format (x0, x1, y0, y1)
        height (int): height of image
        width (int): width of image
    Return:
        List of absolute bounding box values in format (x0, y0, x1, y1)
    """
    x0 = np.round(np.multiply(bbox[0], width))
    x1 = np.round(np.multiply(bbox[1], width))
    y0 = np.round(np.multiply(bbox[2], height))
    y1 = np.round(np.multiply(bbox[3], height))
    return [x0, y0, x1, y1]

def get_detectron_dicts(img_dir):
    """
    Create a Detectron2's standard dataset dicts from an image folder.
    See: https://detectron2.readthedocs.io/tutorials/datasets.html
    
    Args:
        img_dir (str): path to the image folder (train/validation)
    Return:
        dataset_dicts (list[dict]): List of annotation dictionaries for Detectron2.
    """
    
    # Load annotation DataFrame
    mode = "small-" + img_dir if (debug and img_dir=='train') else img_dir
    annot_df = pd.read_csv(f"{mode}-annotations-bbox-target.csv")
    
    # Get all images in `img_dir`
    img_ids = annot_df["ImageID"].unique().tolist()
    img_paths = [f'{img_dir}/{img_id}.jpg' for img_id in img_ids]
    # img_paths = glob.glob(f'{img_dir}/*.jpg')
    
    dataset_dicts = []
    for path in tqdm(img_paths):
        file_name = path
        height, width = cv2.imread(file_name).shape[:2]
        # Get image id from file_name
        img_id = re.findall(f"{img_dir}/(.*).jpg", file_name)[0]
            
        record = {}
        record['file_name'] = file_name
        record['image_id'] = img_id
        record['height'] = height
        record['width'] = width
        
        # Extract bboxes from annotation file
        bboxes = annot_df[['ClassID', 'XMin', 'XMax', 'YMin','YMax']][annot_df['ImageID'] == img_id].values
        annots = []
        for bbox in bboxes:
            # Calculate absolute bounding box
            abs_bbox = bbox_rel_to_abs(bbox[1:], height, width)
            annot = {
                "bbox": abs_bbox,
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(bbox[0]),
            }
            annots.append(annot)

        record["annotations"] = annots
        dataset_dicts.append(record)
    return dataset_dicts


# Register Datasets
from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "validation"]:
    dataset_name = "musical_instruments_" + d
    print("Registering ", dataset_name)
    DatasetCatalog.register(dataset_name, lambda d=d: get_detectron_dicts(d))
    MetadataCatalog.get(dataset_name).set(thing_classes=target_classes)

detectron_metadata = MetadataCatalog.get("musical_instruments_train")


# Config training
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

if os.path.exists(experiment):
    shutil.rmtree(experiment)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{MODEL}.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{MODEL}.yaml")
cfg.DATASETS.TRAIN = ("musical_instruments_train",)
cfg.DATASETS.TEST = ("musical_instruments_validation",)

# Training hyperparameters
cfg.SOLVER.IMS_PER_BATCH = IMS_PER_BATCH
cfg.SOLVER.BASE_LR = BASE_LR
cfg.SOLVER.MAX_ITER = MAX_ITER
cfg.SOLVER.WARMUP_ITERS = WARMUP_ITERS
cfg.SOLVER.GAMMA = GAMMA
cfg.SOLVER.STEPS = STEPS

cfg.DATALOADER.NUM_WORKERS = 6
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(target_classes)
cfg.MODEL.RETINANET.NUM_CLASSES = len(target_classes)
cfg.OUTPUT_DIR = experiment

# Save config to file
with open(experiment + "/config.yaml", "w") as f:
    f.write(cfg.dump())

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)

# Train
trainer.train()

# Evaluate
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("musical_instruments_validation", cfg, False, output_dir=cfg.OUTPUT_DIR)
trainer.test(cfg=cfg,
             model=trainer.model,
             evaluators=evaluator)