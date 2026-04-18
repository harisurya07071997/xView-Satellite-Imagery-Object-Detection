
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# bash Miniconda3-latest-Linux-x86_64.sh
# source ~/.bashrc
# conda create -n pytorch python=3.11
# conda activate pytorch

# pip install ultralytics wandb gpustat
# wandb login 7d3d4da545e2548454d463fc55f1af1409db7195

import os
import gc
import yaml
import torch
import wandb
import shutil
import random
import argparse
import numpy as np
from ultralytics import YOLO
from functools import partial

import warnings
warnings.filterwarnings('ignore')


def clear_gpu_memory():
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✓ Cleared CUDA GPU memory")

    elif torch.mps.is_available():
        torch.mps.empty_cache()
        print("✓ Cleared MPS memory")
    
    else: print("✓ No GPU/MPS device available to clear")
        

def set_seed(seed: int):
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    print(f"✓ Reproducibility enabled with seed: {seed}")


class YOLOTrainer:

    def __init__(self, config: dict):

        self.results = None
        self.best_metrics = None
        self.config = config
        self.model = self.initialize_model()


    def initialize_model(self) -> YOLO:

        if self.config["model"]["yolo"]["resume"]:
            checkpoint_path = self.config["model"]["checkpoint"]
            print(f"Initializing model from checkpoint: {checkpoint_path}")
            return YOLO(checkpoint_path)
        else:
            model_name = self.config["model"]["name"]
            print(f"Initializing pre-trained YOLO model: {model_name}")
            return YOLO(model_name)

    
    def load_best_model(self):

        if self.results is None:
            raise ValueError("Training results are not available. Run train_and_validate() first!")
        
        best_model_path = f"{self.results.save_dir}/weights/best.pt"
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Best model not found at: {best_model_path}")
        
        self.model = YOLO(best_model_path)
        print(f"✓ Best model loaded successfully")

    
    def _setup_callbacks(self):

        def on_train_epoch_end(trainer):
            try:
                if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                    loss_metrics = {
                        'train/box_loss': float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                        'train/cls_loss': float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                        'train/dfl_loss': float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0,
                    }
                    wandb.log(loss_metrics)
            
            except Exception as e:
                error_msg = f"Error logging training losses at epoch {trainer.epoch}: {e}"
                print(f"⚠️  {error_msg}")
        

        def on_val_start(validator):
            print(f"Starting validation on {validator.dataloader.dataset.data['path']}!")


        def on_val_end(validator):
            print("✓ Validation complete!")


        def on_fit_epoch_end(trainer):
            
            try:
                epoch_metrics = {'epoch': trainer.epoch}
                metrics = trainer.metrics if hasattr(trainer, 'metrics') else {}
                has_validation_metrics = any(key.startswith('metrics/') for key in metrics.keys())
                if not has_validation_metrics:
                    raise ValueError(
                        f"No validation metrics found for epoch {trainer.epoch}. Ensure 'val: True' and valid dataset.")
                
                recall = None
                precision = None
                val_metrics_map = {
                    'val/mAP50': 'metrics/mAP50(B)',
                    'val/mAP50-95': 'metrics/mAP50-95(B)',
                    'val/precision': 'metrics/precision(B)',
                    'val/recall': 'metrics/recall(B)'
                }
                for log_key, metric_key in val_metrics_map.items():
                    if metric_key in metrics:
                        value = float(metrics[metric_key])
                        epoch_metrics[log_key] = value
                        if   log_key == 'val/precision': precision = value
                        elif log_key == 'val/recall'   : recall = value
                    else:
                        print(f"⚠️ Warning: Metric {metric_key} not found in validation results")
                        
                if precision is not None and recall is not None:
                    if precision + recall > 0:
                        f1_score = 2 * (precision * recall) / (precision + recall)
                        epoch_metrics['val/f1_score'] = f1_score
                        print(f"📊 Epoch {trainer.epoch} - F1: {f1_score:.4f} (P: {precision:.4f}, R: {recall:.4f})")
                    else:
                        epoch_metrics['val/f1_score'] = 0.0
                        print(f"⚠️ Warning: Both precision and recall are 0, F1 set to 0")
                else:
                    print(f"⚠️ Warning: Could not calculate F1 score - precision or recall missing")
                
                if hasattr(trainer, 'optimizer') and trainer.optimizer:
                    epoch_metrics['train/lr'] = trainer.optimizer.param_groups[0]['lr']
                
                wandb.log(epoch_metrics)
                
            except Exception as e:
                error_msg = f"Error logging validation metrics at epoch {trainer.epoch}: {e}"
                print(f"⚠️  {error_msg}")
                wandb.log({"errors/callback_failure": error_msg})
        
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)
        self.model.add_callback("on_val_start", on_val_start)
        self.model.add_callback("on_val_end", on_val_end)
        self.model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


    def train(self) -> None:

        print("\n" + "=" * 60)
        print("STARTING TRAINING RUN")
        print("=" * 60)
        print(f'Epochs: {self.config["model"]["yolo"]["epochs"]}')
        print(f'Batch Size: {self.config["model"]["yolo"]["batch"]}')
        print(f'Image Size: {self.config["model"]["yolo"]["imgsz"]}')
        print("=" * 60 + "\n")

        if not self.config['model']['yolo']['val']:
            print("⚠️ WARNING: Validation is disabled! Per-epoch metrics will not be logged.")
            print("Set 'val: True' in config to enable validation.\n")
        else:
            with open(self.config["dataset"]["yaml_file"], 'r') as f:
                dataset_config = yaml.safe_load(f)
            if 'val' not in dataset_config or not dataset_config['val']:
                raise ValueError("Validation set not specified in dataset YAML file")
        
        self._setup_callbacks()
        try:
            self.results = self.model.train(
                data = self.config["dataset"]["yaml_file"], **self.config["model"]["yolo"])
        finally:
            clear_gpu_memory()
            print("✓ Training complete!")
            
            trn_metrics = {}
            print("Logging training metrics to W&B!")
            if hasattr(self.results, 'results_dict'):
                for k, v in self.results.results_dict.items():
                    if isinstance(v, (int, float)):
                        k = k.replace("metrics/", "")
                        trn_metrics[f"trn/{k}"] = v
                print(trn_metrics)
                wandb.log(trn_metrics)


    def validate(self, keep_weights: bool = True) -> None:

        print("\n" + "=" * 60)
        print("VALIDATING MODEL")
        print("=" * 60)

        self.load_best_model()
        print("✓ Best model loaded successfully")
        
        self.model.val(plots = True)
        self._log_validation_plots(self.results.save_dir)
        print("✓ Validation complete!")
        
        if not keep_weights:
            shutil.rmtree(f"{self.results.save_dir}/weights/", ignore_errors = True)


    def _log_validation_plots(self, results_dir: str) -> None:

        print("Logging validation plots to W&B!")
        if not results_dir or not os.path.exists(results_dir):
            print(f"Warning: Results directory not found: {results_dir}")
            return
        
        plot_files = {
            'confusion_matrix': 'confusion_matrix.png',
            'confusion_matrix_normalized': 'confusion_matrix_normalized.png',
            'val_batch0_labels': 'val_batch0_labels.jpg',
            'val_batch0_pred': 'val_batch0_pred.jpg',
            'val_batch1_labels': 'val_batch1_labels.jpg',
            'val_batch1_pred': 'val_batch1_pred.jpg',
            'val_batch2_labels': 'val_batch2_labels.jpg',
            'val_batch2_pred': 'val_batch2_pred.jpg',
            'PR_curve': 'PR_curve.png',
            'P_curve': 'P_curve.png',
            'R_curve': 'R_curve.png',
            'F1_curve': 'F1_curve.png',
            'results': 'results.png',
            'labels': 'labels.jpg',
            'labels_correlogram': 'labels_correlogram.jpg'
        }
        
        for log_name, file_name in plot_files.items():
            file_path = f"{results_dir}/{file_name}"
            try:
                if os.path.exists(file_path): wandb.log({log_name: wandb.Image(file_path)})
                else: print(f"Warning: Plot file {file_path} not found")
            
            except Exception as e: 
                print(f"Warning: Failed to log plot {file_path}: {e}")

    
    def train_and_validate(self, keep_weights: bool = True) -> None:

        self.train()
        self.validate(keep_weights = keep_weights)


    def cleanup(self) -> None:

        try:
            if self.model is not None:
                if hasattr(self.model, 'predictor'): del self.model.predictor
                if hasattr(self.model, 'trainer'  ): del self.model.trainer
                del self.model
            
            self.model = None
            self.results = None
            clear_gpu_memory()
            print("✓ Cleanup complete!")

        except Exception as e:
            print(f"Warning: Cleanup error: {e}")


    def __del__(self) -> None:
        
        try:
            self.cleanup()
        except Exception as e:
            print(f"Warning: Encountered error while cleaning up resources: {e}")


def main(args: argparse.Namespace, config: dict) -> None:

    set_seed(config["model"]["yolo"]["seed"])
    
    if args.do_tuning:
        try:
            wandb_run = wandb.init(project = config["model"]["yolo"]["project"], config = config["model"]["yolo"])
            wandb_run.define_metric("epoch")
            wandb_run.define_metric("train/*", step_metric = "epoch", step_sync = True)
            wandb_run.define_metric("val/*",   step_metric = "epoch", step_sync = True)
            
            if wandb_run is None:
                raise RuntimeError("Failed to initialize W&B run for sweep")
            
            config["model"]["yolo"]["close_mosaic"] = 0
            config["model"]["yolo"]["epochs"] = config["model"]["tuning_epochs"]

            config["model"]["yolo"]["hsv_h"] = wandb_run.config.get("hsv_h", config["model"]["yolo"]["hsv_h"])
            config["model"]["yolo"]["hsv_s"] = wandb_run.config.get("hsv_s", config["model"]["yolo"]["hsv_s"])
            config["model"]["yolo"]["hsv_v"] = wandb_run.config.get("hsv_v", config["model"]["yolo"]["hsv_v"])

            config["model"]["yolo"]["translate"] = wandb_run.config.get("translate", config["model"]["yolo"]["translate"])
            config["model"]["yolo"]["scale"] = wandb_run.config.get("scale", config["model"]["yolo"]["scale"])
            
            config["model"]["yolo"]["mixup" ] = wandb_run.config.get("mixup" , config["model"]["yolo"]["mixup" ])
            config["model"]["yolo"]["cutmix"] = wandb_run.config.get("cutmix", config["model"]["yolo"]["cutmix"])

            config["model"]["yolo"]["box"] = wandb_run.config.get("box", config["model"]["yolo"]["box"])
            config["model"]["yolo"]["cls"] = wandb_run.config.get("cls", config["model"]["yolo"]["cls"])
            config["model"]["yolo"]["dfl"] = wandb_run.config.get("dfl", config["model"]["yolo"]["dfl"])

            config["model"]["yolo"]["lr0"] = wandb_run.config.get("lr0", config["model"]["yolo"]["lr0"])
            config["model"]["yolo"]["cos_lr"] = wandb_run.config.get("cos_lr", config["model"]["yolo"]["cos_lr"])

            print(f"Doing hyperparameter tuning with configuration: {wandb_run.config}")

        except Exception as e:
            print(f"Error initializing W&B for sweep: {e}")
            raise
    
    trainer = YOLOTrainer(config)
    try:
        trainer.train_and_validate(keep_weights = args.do_resume_from_checkpoint or args.do_training)
    finally:
        trainer.cleanup()
        if args.do_tuning: wandb.finish()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Quantity Estimation with YOLOv11")
    parser.add_argument("--config_path", type = str, default = "./Config.yaml")
    parser.add_argument("--do_training", action = 'store_true', help = "Flag to trigger training")
    parser.add_argument("--do_resume_from_checkpoint", action = 'store_true', help = "Flag to resume from checkpoint")
    parser.add_argument("--do_tuning", action = 'store_true', help = "Flag to trigger hyperparameter tuning")
    args = parser.parse_args()

    mode_flags = [args.do_tuning, args.do_training, args.do_resume_from_checkpoint]
    if sum(mode_flags) > 1:
        raise ValueError("Only one mode flag can be specified: --do_tuning, --do_training, or --do_resume_from_checkpoint")
    if not any(mode_flags):
        args.do_training = True
    
    if not os.path.exists(args.config_path):
        raise FileNotFoundError(f"Configuration file {args.config_path} not found!")
    
    with open(args.config_path, 'r') as f: config = yaml.safe_load(f)
    config["model"]["yolo"]["close_mosaic"] = int(
        config["model"]["yolo"]["close_mosaic"] * config["model"]["yolo"]["epochs"])
    
    if args.do_resume_from_checkpoint:
        checkpoint_path = config["model"]["checkpoint"]
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise ValueError("Checkpoint path is empty or not found!")
        if not checkpoint_path.endswith(".pt"):
            raise ValueError("Checkpoint file must have .pt extension")
        
        config["model"]["yolo"]["resume"] = True
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        
    project_name = config["model"]["yolo"]["project"]
    try:
        if args.do_tuning:
            sweep_config = config["model"]["wandb"]["sweep_config"]
            sweep_id = wandb.sweep(sweep_config, project = project_name)
            partial_main = partial(main, args, config)
            wandb.agent(sweep_id, partial_main, count = 100)
        else:
            wandb_run = wandb.init(project = project_name, config = config["model"]["yolo"])
            wandb_run.define_metric("epoch")
            wandb_run.define_metric("train/*", step_metric = "epoch", step_sync = True)
            wandb_run.define_metric("val/*",   step_metric = "epoch", step_sync = True)
            main(args = args, config = config)
    
    except Exception as e:
        print(f"Error in main execution: {e}")
        raise

    finally:
        wandb.finish()
        clear_gpu_memory()