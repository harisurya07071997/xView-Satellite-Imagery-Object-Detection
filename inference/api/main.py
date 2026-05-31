# FastAPI Inference Backend -- to serve the model inference request
# handles the model loading, image preprocessing, and inference (prediction) logic
import io
import os
import time
import shutil
import logging
import s3_utils
import traceback
import numpy as np
from PIL import Image
from pathlib import Path
from app import Inference
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


BASE_DIR = Path(__file__).resolve().parent



class PredictionService:
    def __init__(self, config_path, model_path=None):
        self.detector= Inference(config_path= config_path, model_path= model_path)

        # initialize output directory to store the annotated image
        self.output_dir= BASE_DIR / "outputs" / "annotated"
        os.makedirs(self.output_dir, exist_ok=True)

    # Pre-Process
    async def preprocess(self, file: UploadFile):
        
        start= time.time()
        contents= await file.read()

        # read image
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            self.save_file_name= file.filename
        except Exception:
            raise ValueError("Invalid image file")

        preprocess_time= time.time() - start

        return {
            "image": image,
            "metadata": {
                "filename": file.filename,
                "image_size": image.size,
            },
            "latency": {
                "preprocess_time": round(preprocess_time, 3)
            }
        }
    
    # Process
    def process(self, preprocess_output):

        start= time.time()
        # run the inference
        predictions, latency_dict= self.detector.run_inference(preprocess_output["image"])
        process_time= time.time() - start

        return {
            "predictions": predictions,
            "metadata": {
                **preprocess_output["metadata"],
                "num_predictions": len(predictions),
            },
            "latency": {
                **preprocess_output["latency"],
                "process_time": round(process_time,3),
                **latency_dict,
            }
        }
    
    # Post-Process
    def postprocess(self, preprocess_output, process_output):

        start= time.time()

        # save annotated image
        annotation_save_path= self.save_annotated_image(preprocess_output["image"], process_output["predictions"])

        # upload annotated image in s3 bucket
        s3_url= s3_utils.upload_annotated_image_to_s3(annotation_save_path)

        # remove uploaded image in local disk
        Path(annotation_save_path).unlink(missing_ok=True)

        response= {"success": True,
                   "result_image_url": s3_url,
                   "metadata": process_output["metadata"],
                   "latency": process_output["latency"],
                   "predictions": process_output["predictions"]}
        
        postprocess_time= time.time() - start

        response["latency"]["postprocess_time"]= round(postprocess_time, 3)

        return response

    
    def save_annotated_image(self, image, predictions):

        annotated_image= self.detector.draw_predictions(image, predictions)

        # unique filename
        filename = f"annotated_{self.save_file_name}.jpg"

        save_path= self.output_dir / filename
       
        annotated_image.save(save_path)
        logging.info(f"Annotated Image Saved Successfully in {save_path}")

        return save_path


    
    def warmup(self):
        
        logging.info("Running Model Warmup...")

        dummy= np.zeros((512,512,3), dtype= np.uint8)
        
        self.detector.run_inference(image= dummy)

        logging.info("Warmup Completed.")



# ****** FastAPI *******
app = FastAPI()

@app.on_event("startup")
def startup_event():

    try:

        model_path = s3_utils.download_model_from_s3()

        app.state.service = PredictionService(config_path=BASE_DIR / "Config.yaml", model_path=model_path)

        app.state.service.warmup()

        print("Server ready.")

    except Exception as e:

        print(f"Startup failed: {e}")

        traceback.print_exc()

        raise e


@app.get("/health")
def health():
    return {"status": "healthy",
            "model_loaded": hasattr(app.state, "service")}

@app.post("/predict")
async def predict(request: Request, file: UploadFile= File(...)):

    try:
        logging.info("Inference Started")

        pipeline_start = time.time()

        service = request.app.state.service

        # preprocess
        preprocess_output= await service.preprocess(file)
        # process
        process_output = await run_in_threadpool(service.process,
                                                 preprocess_output)
        
        # postprocess
        final_output= service.postprocess(preprocess_output, process_output)

        logging.info("Inference Ended")
        
        total_pipeline_time= time.time() - pipeline_start
        final_output["latency"]["total_pipeline_time"]= round(total_pipeline_time, 3)

        return JSONResponse(status_code= 200,
                            content= final_output)
    
    except Exception as e:
        traceback.print_exc()

        return JSONResponse(status_code= 500,
                            content= {"success": False,
                                      "error": str(e)}
                            )

# kill -9 <PID>
# lsof -i :8000
# docker run -d -p 8000:8000 xview-api