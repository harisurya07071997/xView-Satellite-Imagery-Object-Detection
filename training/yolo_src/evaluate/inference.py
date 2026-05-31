from ultralytics import YOLO
import glob
import os
import pickle 

def load_model(model_path):
    
    model= YOLO(model_path)
    print(f"Model has been loaded!")
    return model

model= load_model(model_path='./model.pt')


def prepare_evaluation_data(image_files, conf=0.3, iou=0.5, imgsz=512):
    gt_boxes = {}
    pred_boxes = {}

    output_dir = "../output/Inference-Results"
    os.makedirs(output_dir, exist_ok=True)

    results = model.predict(image_files,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            batch=16,   
                            stream=True,
                            verbose=False,
                            device= 0
                           )
    
    for im_file, result in zip(image_files, results):

        image_id = os.path.splitext(os.path.basename(im_file))[0]

        # ---------------- GT ----------------
        gt_data = []
        label_file = im_file.replace("images", "labels").replace(".png", ".txt")

        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    values = line.strip().split()
                    gt_label = int(values[0])
                    x, y, w, h = map(float, values[1:])
                    gt_data.append({
                        "bbox": [x, y, w, h],
                        "class": gt_label
                    })

        gt_boxes[image_id] = gt_data

        # ---------------- PRED ----------------
        #result = model.predict(im_file, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]

        pred_data = []

        if result.boxes is not None and len(result.boxes) > 0:
            bboxes = result.boxes.xywhn.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()  

            for bbox, c, score in zip(bboxes, cls, scores):
                pred_data.append({
                    "bbox": bbox.tolist(),
                    "class": int(c.item()),
                    "score": float(score)
                })

        pred_boxes[image_id] = pred_data

    with open(f"{output_dir}/preds.pkl", "wb") as f:
        pickle.dump(pred_boxes, f)

    with open(f"{output_dir}/gt.pkl", "wb") as f:
        pickle.dump(gt_boxes, f)

    print(f"Successfully saved pkl files in '{output_dir}'")

if __name__ == "__main__":

    files= sorted(glob.glob("../dataset/Processed/images/val/*.png", recursive=True))
    labels= sorted(glob.glob("../dataset/Processed/labels/val/*.txt", recursive=True))

    assert len(files) == len(labels)
    print(f"Total files found: {len(files)}, Total labels found: {len(labels)}")

    model= load_model(model_path='../output/model/best.pt')
    prepare_evaluation_data(files, conf= 0.3, iou=0.5, imgsz= 512)
    exit(0)    