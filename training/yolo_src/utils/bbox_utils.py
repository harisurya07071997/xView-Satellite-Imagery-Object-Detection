import pandas as pd
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw
import random


#------------------------------------------------#
# Export Annotations and save it in a directory
#------------------------------------------------#

def export_to_yolo(annot_df, output_dir, tile_size):
    """
    Convert annotations from a DataFrame into YOLO format and save them as text files.

    Parameters
    ----------
    annot_df : pandas.DataFrame
        DataFrame containing annotation data. Expected columns include:
        - 'tile_image_id'
        - 'label' or class ID
        - 'split'
        - Bounding box coordinates (e.g., xmin, ymin, xmax, ymax)

    output_dir : str
        Directory where YOLO annotation `.txt` files will be saved.
        One file will be created per image.

    tile_size : int

    Returns
    -------
    None
        Writes YOLO annotation files to the specified output directory.

    """
    
    os.makedirs(output_dir, exist_ok=True)

    for name, group in annot_df.groupby("tile_image_id"):

        # Extract values
        bboxes = group[["xmin","ymin","xmax","ymax"]].values.astype(np.float32)
        class_ids = group["label"].values.astype(int)
        split= np.unique(group["split"].values)
        if len(split) > 1: raise ValueError(f"Invalid Split. tile_image_id: {name} in dataframe has multiple splits for an image")

        # Convert to YOLO format
        x_center = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        y_center = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        width    = (bboxes[:, 2] - bboxes[:, 0])
        height   = (bboxes[:, 3] - bboxes[:, 1])

        # Normalize
        x_center /= tile_size
        y_center /= tile_size
        width    /= tile_size
        height   /= tile_size

        # Create txt file name
        txt_name = os.path.splitext(name)[0] + ".txt"
        os.makedirs(os.path.join(output_dir, split.item()), exist_ok=True)
        txt_path = os.path.join(output_dir, split.item(), txt_name)

        # Write to file
        with open(txt_path, "w") as f:
            for cls, xc, yc, w, h in zip(class_ids, x_center, y_center, width, height):
                f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    print("YOLO txt exported successfully.")



def export_to_coco(annot_df, output_dir, tile_size):
    """
    Convert annotations from a DataFrame into COCO format and save them in a single json file.

    Parameters
    ----------
    annot_df : pandas.DataFrame
        DataFrame containing annotation data. Expected columns include:
        - 'tile_image_id'
        - 'label' or class ID
        - 'split'
        - Bounding box coordinates (e.g., xmin, ymin, xmax, ymax)

    output_dir : str
        Directory where COCO annotation `.json` files will be saved.

    tile_size : int

    Returns
    -------
    None
        Writes YOLO annotation files to the specified output directory.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    coco_json= {
        "info": {
        "description": "xView Satellite Imagery Dataset (Tiled)",
        "date_created": datetime.now().isoformat(),
        },
        "images": [],
        "annotations": [],
        "categories": []
    }

    # store categories metadata
    categories= annot_df[["label", "category"]].drop_duplicates().sort_values("label")
    for row in categories.itertuples(index=False):
        coco_json["categories"].append({
            "id": int(row.label),
            "name": row.category,
            "supercategory": "none"
        })

    # store images metadata
    image2id= {}
    images= []
    unique_tiles = annot_df["tile_image_id"].unique()
    for i, tile_image_id in enumerate(unique_tiles):
        images.append({"id": i,
                       "file_name": tile_image_id,
                       "width": tile_size,
                       "height": tile_size
                      })
        image2id[tile_image_id]= i

    coco_json["images"]= images
    
    # store annotations metadata
    annot_id= 0
    annotations = []
    for row in annot_df.itertuples(index=False):
        xmin = row.xmin
        ymin = row.ymin
        xmax = row.xmax
        ymax = row.ymax

        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue
        area = width * height
        
        tile_image_id= row.tile_image_id
        category_id= row.label

        annotations.append({"id": annot_id,
                            "image_id": image2id[tile_image_id],
                            "category_id": int(category_id),
                            "bbox": [xmin, ymin, width, height],
                            "area": area,
                            "segmentation": [],
                            "iscrowd": 0
                           })
        annot_id+=1
        
    coco_json["annotations"]= annotations    
    
    # 5. Export to JSON
    output_path= os.path.join(output_dir, "annotations.json")
    with open(output_path, 'w') as f:
        json.dump(coco_json, f, indent=4)
    
    print("COCO JSON exported successfully.")
    return coco_json




#------------------------------------------------#
# Annotations Visualization
#------------------------------------------------#

def visualize_yolo_annotation(image_dir, label_dir, image_path=None):
    """
    Visualize YOLO-format annotations by overlaying bounding boxes on an image.
   
    Parameters
    ----------
    image_dir : str
        Path to the directory containing input images.

    label_dir : str
        Path to the directory containing YOLO annotation files (.txt format).

    image_path : str, optional (default=None)
        Specific image file path to visualize. If None, a random image from
        image_dir will be selected.

    Returns
    -------
    PIL.Image
        Image with bounding boxes drawn on it.
    """

    # random image if not provided
    if image_path is None:
        images = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        image_name = random.choice(images)
        image_path = os.path.join(image_dir, image_name)
    else:
        image_name = os.path.basename(image_path)

    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    draw = ImageDraw.Draw(img)

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    if not os.path.exists(label_path):
        print(f"No label file found for {image_name}")
        return img

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, bw, bh = map(float, line.strip().split())

        xc *= w
        yc *= h
        bw *= w
        bh *= h

        xmin = int(xc - bw / 2)
        ymin = int(yc - bh / 2)
        xmax = int(xc + bw / 2)
        ymax = int(yc + bh / 2)

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), str(int(cls)), fill="yellow")

    return img



def visualize_coco_annotation(image_dir, out_dir, coco_json_path, image_id=None):
    """
    Visualize COCO format annotations by overlaying bounding boxes on an image.
   
    Parameters
    ----------
    image_dir : str
        Path to the directory containing input images.

    coco_json_path : str
        Path to the file containing YOLO annotation files (.txt format).

    image_id : str, optional (default=None)
        Specific image file id to visualize. If None, a random image from
        image_dir will be selected.

    Returns
    -------
    PIL.Image
        Image with bounding boxes drawn on it.
    """

    # load json  
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # fetch images and annotations
    images = coco["images"]
    annotations = coco["annotations"]

    if image_id is None:
        img_info = random.choice(images)
    else:
        img_info = next(img for img in images if img["id"] == image_id)

    
    file_name = img_info["file_name"]
    image_id = img_info["id"]

    image_path = os.path.join(image_dir, file_name)
    img = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(img)

    anns = [ann for ann in annotations if ann["image_id"] == image_id]

    for ann in anns:
        x, y, w, h = ann["bbox"]

        xmin = int(x)
        ymin = int(y)
        xmax = int(x + w)
        ymax = int(y + h)

        # Draw rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        draw.text((xmin, ymin), str(ann["category_id"]), fill="yellow")

    return img


