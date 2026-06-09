# Multi-Task Deep Learning Framework for Ultrasound Image Classification & Semantic Segmentation

An end-to-end medical computer vision repository engineering Keras 3 layers mapped directly onto a stable PyTorch execution backend engine. The architecture processes multi-class classification configurations (Benign, Malignant, Normal) and applies self-attention semantic segmentation overlays to track lesion boundaries inside breast ultrasound image directories. Optimized to ensure hardware loop stability on consumer GPU layers (NVIDIA RTX 3050 via WSL2 Ubuntu virtual environments).

## 🚀 Repository Pipeline & Cell Breakdown

### Cell 1: Environment Setup & Keras Multi-Backend Engine Routing
* **Purpose**: Initializes the hardware environment constraints and locks global reproducible random seeds across frameworks.
* **Key Operations**: Maps `os.environ["KERAS_BACKEND"] = "torch"` to route standard Keras APIs through native PyTorch backend layers. It configures `torch.backends.cudnn.enabled = False` to prevent native cuDNN compilation trace crashes inside WSL systems. It also includes an automated garbage collection buffer tool (`flush_system_memory()`) to prevent VRAM memory overflows.
* **Core Classes**: 
  * `ProjectConfig`: Encapsulates spatial input bounds (`224x224`), dataset anchors (`/home/mamdouh_salem/ultrasound_data`), and training hyper-parameters.

### Cell 2: Instructor Data Augmentation & Universal Model Generators
* **Purpose**: Coordinates structural data matrix splits and manages live spatial data augmentation streaming pipelines.
* **Key Operations**: Generates a balanced 80/10/10 data division matrix using a strict stratified profile (`train_test_split(stratify=...)`) to correct underlying medical class imbalances. It implements a custom NumPy filter (`data_augment_np`) that executes random rotations, mirror flips, and brightness variance via OpenCV without degrading physical medical tissue borders.
* **Core Classes**:
  * `UltrasoundGenerator(keras.utils.PyDataset)`: Streams data batches on the fly, scales pixel intensities down between `0.0` and `1.0`, and yields stable one-hot encoded matrix targets.

### Cell 3: Medical Mask Segmentation Generator Stream
* **Purpose**: Establishes parallel high-density target stream arrays specifically for self-attention semantic mapping workflows.
* **Key Operations**: Loads source scans alongside corresponding target segmentation masks. It enforces a crisp mass contour boundary cutoff during resizing operations by strictly using nearest-neighbor interpolation (`cv2.INTER_NEAREST`) and standard thresholding arrays (`np.where(mask > 0.5, 1.0, 0.0)`).
* **Core Classes**:
  * `UltrasoundSegmentationDataset(keras.utils.PyDataset)`: Pushes pixel-level validation coordinates, utilizing channel expansion tools (`np.expand_dims(..., axis=-1)`) to handle multi-dimensional loss calculation shapes.

### Cell 4: Model Architecture Build Factory
* **Purpose**: Acts as the centralized structural production engine compiling custom, convolutional, and transformer-based model layouts.
* **Key Operations**: Integrates explicit tensor permutations (`layers.Permute`) to switch incoming channel-last images into channel-first inputs required by HuggingFace Transformers. It drops leaky PyTorch functional lambdas, replacing them with standard Keras `layers.UpSampling2D(size=(4, 4))` layers to resize downscaled matrices safely.
* **Core Classes**:
  * `ViTBackboneLayer(layers.Layer)`: Wraps a pre-trained Vision Transformer model configuration (`google/vit-base-patch16-224`).
  * `SegFormerBackboneLayer(layers.Layer)`: Wraps a pre-trained NVIDIA `SegformerForSemanticSegmentation` layer (`nvidia/mit-b0`), returning pure backpropagatable tensors.
  * `MedicalModelFactory`: Composes production pipelines for Custom CNNs from scratch, frozen Feature Extraction convolutional loops (`ResNet50`), global attention classifiers, and custom pixel-wise segmentation models.

### Cell 5: Performance Evaluation Engine and Mask Visualizer
* **Purpose**: Evaluates cross-validation performance parameters and renders high-fidelity comparative visual arrays.
* **Key Operations**: Tracks classification reports containing precision, recall, and F1-score matrices. It configures Matplotlib graphics to render triple-column verification charts comparing source scans, ground truth targets, and predicted attention paths.
* **Core Classes**:
  * `PerformanceEvaluator`: Handles automated testing routines. It pulls explicit batch array index targets (`images, masks = seg_test_gen`) to process visualization graphs cleanly without encountering Python generator unpacking trace errors.

### Cell 6: Classification Master Execution Loop
* **Purpose**: Manages and triggers sequential model training phases across all primary image classification setups.
* **Key Operations**: Coordinates consecutive training workflows across Phase 1 (Baseline CNN, lr=1e-4), Phase 2 (ResNet-50 Transfer Learning with frozen convolutional feature blocks, lr=1e-5), and Phase 3 (HuggingFace ViT using a low learning rate anchor of 2e-5). It runs explicit VRAM flushes between models to guarantee runtime safety.

### Cell 7: Segmentation Task Run Engine (Hugging Face SegFormer)
* **Purpose**: Fine-tunes and validates the final self-attention semantic mapping model to trace target lesion areas.
* **Key Operations**: Compiles the `SegFormer` layout with Binary Cross-Entropy loss to evaluate errors pixel-by-pixel, executes training loops on mask pipelines, and renders final overlay graphics for presentations.
