
## Inference Pipeline Overview

This section describes the end-to-end inference workflow designed for large-scale satellite imagery object detection using a production-oriented deployment architecture.

Unlike standard object detection pipelines, satellite imagery inference introduces several engineering challenges:
- extremely high-resolution images,
- GPU memory limitations,
- dense object distributions,
- and small-object detection sensitivity.

To address these challenges, the inference pipeline was designed with:
- tile-based image processing,
- efficient batching,
- coordinate remapping,
- global postprocessing,
- and scalable API deployment.

The pipeline supports:
- large-image inference without aggressive resizing,
- GPU-efficient processing through image tiling,
- merging predictions from multiple tiles,
- and cloud-ready deployment using FastAPI, Docker, AWS EC2, and Amazon S3.

The inference pipeline was designed to bridge the gap between research experimentation and real-world deployment by enabling scalable, production-grade object detection for high-resolution satellite imagery.

