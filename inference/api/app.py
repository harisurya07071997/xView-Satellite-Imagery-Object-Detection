# Model Inference Pipeline

#import libraries
import yaml
import time
import torch
import logging
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
from PIL import Image, ImageDraw


class Inference:
    def __init__(self, config_path, model_path=None):
        # load config 
        with open(config_path, 'r') as f: 
            self.config = yaml.safe_load(f)

        if model_path is None:
            model_path = self.config["model"]["model_path"]

        # load model
        self.model= self.initialize_model(model_path)  

        # pre-processing parameters
        self.tile_size= self.config["model"]["parameters"]["tile_size"]
        self.tile_overlap= self.config["model"]["parameters"]["tile_overlap"]  

        # model parameters
        self.batch= self.config["model"]["parameters"]["batch_size"]  
        self.iou_thresh= self.config["model"]["parameters"]["iou_thresh"]  
        self.conf_thresh= self.config["model"]["parameters"]["conf_thresh"]  

        # post-processing parameters
        self.nms_thresh= self.config["model"]["parameters"]["nms_thresh"]  


    def initialize_model(self, path):
        """
        YOLO Model Initialization
        """

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Running inference on device: {self.device}")
        
        model= YOLO(path)
        model.to(self.device)
        logging.info(f"Model Initialization Succeeded")
        
        return model
    

    #*********************Pre-Processing***************************

    def get_positions(self, dim_size, stride):
        """
        Generate uniform tile positions using padding strategy.

        Keeps constant stride spacing and allows final tile
        to exceed image boundary (requires external padding).
        """

        # image smaller than tile
        if dim_size <= self.tile_size:
            return [0]

        # generate positions with uniform stride
        positions = np.arange(
            0,
            dim_size,
            stride,
            dtype=np.int32
        )

        return positions.tolist()
    
    
    def get_tile_image(self, image_arr):
        """
        Generate tiled image patches with 'REFLECT' padding.
        """

        # global image size
        h,w,= image_arr.shape[:2]
        # overlap steps
        stride= int(self.tile_size * (1 - self.tile_overlap))
        stride= max(1, stride)

        # pre-computation of x and y tile positions
        y_positions= self.get_positions(h, stride)
        x_positions= self.get_positions(w, stride)

        tiles= []

        for y in y_positions:
            for x in x_positions:

                # image slicing
                tile= image_arr[y: y+self.tile_size, x: x+self.tile_size]
                orig_h, orig_w = tile.shape[:2]

                # within the boundary
                if orig_h == self.tile_size and orig_w == self.tile_size:
                    padded_tile = tile

                # exceeding boundary 
                else:
                    # fetch pad regions
                    pad_y= self.tile_size - orig_h
                    pad_x= self.tile_size - orig_w

                    # reflect pading
                    padded_tile = np.pad(tile,((0, pad_y),(0, pad_x),(0, 0)),
                                        mode="reflect")

                tiles.append((padded_tile,
                            (x, y),
                            (orig_h, orig_w)))
        
        return tiles
    
    #**************************Post-Processing***************************

    def apply_global_nms(self, all_predictions):
        """
        Class-wise Non-maximum Suppression (NMS) over global coordinates to suppress same objects which got detected multiple times during tile overlap
        """

        if len(all_predictions) == 0:
            return []

        # fetch boxes, cls, scores separately
        boxes= torch.tensor([p['bbox'] for p in all_predictions], dtype= torch.float32)
        classes= torch.tensor([p['class_id'] for p in all_predictions])
        scores= torch.tensor([p['score'] for p in all_predictions], dtype= torch.float32)

        final_predictions= []

        unique_classes= classes.unique()
        for cls_id in unique_classes:
            
            # fetch class-wise predictions
            cls_mask= classes == cls_id
            cls_boxes= boxes[cls_mask]
            cls_scores= scores[cls_mask]

            # non-maximum suppression based on boxes and scores
            keep= nms(boxes= cls_boxes, scores= cls_scores, iou_threshold= self.nms_thresh)

            # fetch indexes of all predictions with the corresponding classes that passes nms
            cls_predictions= [all_predictions[i] for i in torch.where(cls_mask)[0][keep]]

            final_predictions.extend(cls_predictions)

        return final_predictions
    

    #*******************************Inference Pipeline******************************************

    def run_inference(self, image):
        """
        Tile Based Batch Inference
        """
    
        # load image
        #image= Image.open(img_path)
        image_arr= np.array(image)

        # image tiling
        start= time.perf_counter()
        image_tiles_metadata= self.get_tile_image(image_arr)
        tile_generation_time= time.perf_counter() - start

        # batch inference
        im_tiles= [tile[0] for tile in image_tiles_metadata] 
        logging.info(f"Tiles Generated! Number of tiles: {len(im_tiles)}")
        
        start= time.perf_counter()

        with torch.inference_mode():
            results= self.model.predict(im_tiles,
                                imgsz= self.tile_size,
                                conf= self.conf_thresh,
                                iou= self.iou_thresh,
                                batch= self.batch,
                                half= self.device == "cuda",
                                verbose=False
                                )
        model_inference_time= time.perf_counter() - start

        # class_id -> class_name mapping
        class_map= self.model.names
        
        
        all_predictions = []
        
        start= time.perf_counter()
        for tile_info, result in zip(image_tiles_metadata, results):
        
            tile_x, tile_y = tile_info[1]
            orig_h, orig_w= tile_info[2]
            
            # fetch box details
            bboxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            for bbox, cls_id, score in zip(bboxes, classes, scores):
                # convert box coords from xywhn -> xyxy
                x1,y1,x2,y2= bbox
                
                # remove detections from padded area
                if x1 >= orig_w or y1 >= orig_h:
                    continue

                # clip coords to valid tile region
                x1= max(0, min(x1, orig_w))
                y1= max(0, min(y1, orig_h))
                x2= max(0, min(x2, orig_w))
                y2= max(0, min(y2, orig_h))

                # map tile coords ---> global coords
                x1 += tile_x
                y1 += tile_y
                x2 += tile_x
                y2 += tile_y

                all_predictions.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class_id": int(cls_id),
                    "class_name": class_map[int(cls_id)],
                    "score": float(score),
                    "tile_x": tile_x,
                    "tile_y": tile_y})
                
        coordinate_mapping_time= time.perf_counter() - start

        # global nms to avoid duplicate predictions 
        start= time.perf_counter()
        final_predictions= self.apply_global_nms(all_predictions)
        global_nms_time= time.perf_counter() - start

        latency_dict= {"tile_generation_time": round(tile_generation_time, 4),
                       "model_inference_time": round(model_inference_time, 4),
                       "coordinate_mapping_time": round(coordinate_mapping_time, 4),
                       "global_nms_time": round(global_nms_time, 4)}

        return final_predictions, latency_dict
    

    #**************************** Annotate Predictions****************************

    def draw_predictions(self, image, predictions):
        """
        visualize the predictions
        """

        # if instance is other than PIL Image, convert to PIL
        if not isinstance(image, Image.Image):
            image= Image.fromarray(image)

        draw= ImageDraw.Draw(image)

        for pred in predictions:
            x1,y1,x2,y2= pred['bbox']
            class_name= pred['class_name']
            score= pred['score']

            label= f"{class_name} {score:.2f}"

            # draw bbox
            draw.rectangle([x1,y1,x2,y2], outline="red", width= 2)

            # draw corresponding text
            draw.text((x1, max(0, y1-12)), label, fill="red")

        return image

    