#!/usr/bin/env python3
"""
A consolidated script for face detection, age/gender prediction, activation extraction,
model visualization, and Mantel test analysis. This script preserves the original functionality
of the Jupyter notebook while organizing the code into reusable functions and providing
clear comments and a main guard.
"""

import os
import glob
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple

import cv2
import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn as nn
from onnx2torch import convert
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS
from torchviz import make_dot
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils.face_align import transform


# Assuming utils.config.load_config and rsa.permutation_analysis.run_mantel_test are available
from utils.config import load_config
from rsa.permutation_analysis import run_mantel_test


class FaceModelBase(ABC):
    """
    Abstract base class for face-related models.
    Enforces a common interface across detection and attribute models.
    """

    @abstractmethod
    def prepare(self, *args, **kwargs):
        """
        Optional setup before prediction (e.g., thresholds, input sizes).
        """
        pass

    @abstractmethod
    def predict(
        self, img: np.ndarray, bboxes: List[Tuple[float, float, float, float]]
    ) -> List:
        """
        Predict outputs based on an image and (optional) bounding boxes.

        - For detection models: `bboxes` is ignored; return new bounding boxes.
        - For attribute models: `bboxes` are used to crop/align faces.
        """
        pass


class RetinaFaceTorch(FaceModelBase):
    """
    PyTorch-based port of the ONNXRuntime+NumPy RetinaFace wrapper.

    Usage example:
        rft = RetinaFaceTorch(model_file="retinaface.onnx")
        rft.prepare(nms_thresh=0.4, det_thresh=0.5, input_size=(640, 640))
        dets, kpss = rft.detect(image)  # image is an HxWx3 BGR uint8 numpy array
    """

    def __init__(self, model_file: str, input_size: Tuple[int, int] = (640, 640)):
        assert model_file is not None, "Must provide ONNX model file path"
        assert os.path.exists(model_file), f"Model file not found: {model_file}"

        # Load and convert the ONNX model to a PyTorch model
        onnx_model = onnx.load(model_file)
        self.torch_model = convert(onnx_model)
        self.torch_model.eval()

        # Store some defaults
        self.model_file = model_file
        self.center_cache: dict = {}  # Cache for anchor centers
        self.nms_thresh: float = 0.4  # NMS threshold
        self.det_thresh: float = 0.5  # Detection confidence threshold
        self.input_size: Tuple[int, int] = input_size  # (width, height)
        self.input_mean: float = 127.5  # Mean for blobFromImage
        self.input_std: float = 128.0  # Std for blobFromImage
        self.use_kps: bool = False  # Whether to predict keypoints

        # Inspect the ONNX outputs to determine FPN configuration
        import onnxruntime as ort  # noqa: F401

        ort_sess = ort.InferenceSession(model_file, None)
        out_count = len(ort_sess.get_outputs())
        if out_count in (6, 9):
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
        else:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]

        # Determine number of anchors per location
        # Single anchor by default, increase if necessary
        self._num_anchors = 1
        if out_count in (6, 9):
            self._num_anchors = 2
        if out_count in (9, 15):
            self.use_kps = True

    def prepare(
        self,
        nms_thresh: float = None,
        det_thresh: float = None,
        input_size: Tuple[int, int] = None,
    ):
        """
        Configure thresholds and input canvas size.

        Args:
            nms_thresh: New non-maximum suppression threshold.
            det_thresh: New detection confidence threshold.
            input_size: New input canvas size as (width, height).
        """
        if nms_thresh is not None:
            self.nms_thresh = nms_thresh
        if det_thresh is not None:
            self.det_thresh = det_thresh
        if input_size is not None:
            self.input_size = input_size

    def forward(self, img: np.ndarray) -> List[torch.Tensor]:
        """
        Run a forward pass through the PyTorch model on a preprocessed blob.
        Returns a list of torch.Tensor outputs in the same order as ONNX outputs.

        Args:
            img: Pre-cropped/resized image (HxWx3 uint8 BGR numpy array).

        Returns:
            A tuple or list of torch.Tensor outputs.
        """
        H, W = img.shape[:2]

        # Prepare the blob (scale, mean/std normalization, swap BGR->RGB)
        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=1.0 / self.input_std,
            size=(W, H),
            mean=(self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        inp = torch.from_numpy(blob).float()

        with torch.no_grad():
            outs = self.torch_model(inp)

        # Ensure outputs are a tuple/list
        if isinstance(outs, torch.Tensor):
            outs = (outs,)
        return outs

    def detect(
        self, img: np.ndarray, max_num: int = 0, metric: str = "default"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full detection pipeline: preprocess, forward pass, and postprocess to get bounding boxes and optional keypoints.

        Args:
            img: HxWx3 uint8 BGR image.
            max_num: Maximum number of detections to keep (0 = keep all).
            metric: Metric for ranking detections if max_num > 0.

        Returns:
            det: N×5 numpy array of [x1, y1, x2, y2, score].
            kpss: N×5×2 numpy array of keypoints or None if keypoints are disabled.
        """
        assert img.dtype == np.uint8, "Input must be uint8 BGR image"

        # Resize & letterbox into input_size canvas
        if self.input_size is None:
            in_w, in_h = img.shape[1], img.shape[0]
        else:
            in_w, in_h = self.input_size

        im_ratio = img.shape[0] / img.shape[1]
        model_ratio = in_h / in_w
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = in_w
            new_h = int(new_w * im_ratio)

        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Forward pass
        outs = self.forward(canvas)

        # Prepare lists to collect scores, boxes, and keypoints
        scores_list, bboxes_list, kps_list = [], [], []

        # Split outputs according to FPN structure:
        # - confidences: indices [0..fmc-1]
        # - bbox regressions: indices [fmc..2*fmc-1]
        # - keypoints (if any): indices [2*fmc..]
        for idx, stride in enumerate(self._feat_stride_fpn):
            # Confidence scores for this stride-level
            sc = outs[idx].cpu().numpy().squeeze()

            # Bounding-box predictions for this stride-level
            bd = outs[idx + self.fmc].cpu().numpy() * stride

            # Keypoint predictions if enabled
            if self.use_kps:
                kp = outs[idx + 2 * self.fmc].cpu().numpy() * stride

            # Compute grid size for this stride
            height = canvas.shape[0] // stride
            width = canvas.shape[1] // stride
            key = (height, width, stride)

            # Reuse or compute anchor centers
            if key in self.center_cache:
                centers = self.center_cache[key]
            else:
                # Build a (height × width × 2) array with x and y indices
                ac = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1  # yields [x_grid, y_grid]
                ).astype(np.float32)
                # Scale by stride and flatten to (N, 2)
                ac = (ac * stride).reshape(-1, 2)
                # If multiple anchors per location, replicate centers
                if self._num_anchors > 1:
                    ac = np.stack([ac] * self._num_anchors, axis=1).reshape(-1, 2)
                centers = ac
                self.center_cache[key] = centers

            # Decode bounding boxes: distance2bbox returns [x1, y1, x2, y2]
            from insightface.model_zoo.retinaface import distance2bbox, distance2kps

            bbs = distance2bbox(centers, bd.reshape(-1, 4))

            # Filter boxes by detection threshold
            pos = np.where(sc.ravel() >= self.det_thresh)[0]
            scores_list.append(sc.ravel()[pos])
            bboxes_list.append(bbs[pos])

            if self.use_kps:
                # Decode keypoints and reshape to (N, 5, 2)
                kpss = distance2kps(centers, kp.reshape(-1, kp.shape[-1]))
                kpss = kpss.reshape(-1, kp.shape[-1] // 2, 2)
                kps_list.append(kpss[pos])

        # Stack all strides together
        scores = np.vstack([s.reshape(-1, 1) for s in scores_list])
        bboxes = np.vstack(bboxes_list)
        kpss = np.vstack(kps_list) if self.use_kps else None

        # Sort detections by confidence score (descending)
        order = scores[:, 0].argsort()[::-1]
        pre = np.hstack((bboxes, scores)).astype(np.float32)
        pre = pre[order]
        if kpss is not None:
            kpss = kpss[order]

        # Apply Non-Maximum Suppression (NMS)
        keep = self.nms(pre)
        det = pre[keep]
        if kpss is not None:
            kpss = kpss[keep]

        # Rescale boxes and keypoints to original image dimensions
        scale = (
            (new_h / img.shape[0])
            if (im_ratio > model_ratio)
            else (new_w / img.shape[1])
        )
        det[:, :4] /= scale
        if kpss is not None:
            kpss /= scale

        # If a max number is specified, keep only the top-scoring detections by a metric
        if max_num > 0 and len(det) > max_num:
            areas = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            centers_img = np.vstack(
                [(det[:, 0] + det[:, 2]) / 2, (det[:, 1] + det[:, 3]) / 2]
            ).T
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
            dists = np.sum((centers_img - img_center) ** 2, axis=-1)
            if metric == "default":
                metrics = areas - 2 * dists
            else:
                metrics = areas
            idxs = np.argsort(metrics)[::-1][:max_num]
            det = det[idxs]
            if kpss is not None:
                kpss = kpss[idxs]

        return det, kpss

    def nms(self, dets: np.ndarray) -> List[int]:
        """
        Pure NumPy Non-Maximum Suppression (NMS).

        Args:
            dets: N×5 array of [x1, y1, x2, y2, score].

        Returns:
            List of indices to keep.
        """
        x1, y1, x2, y2, scores = (
            dets[:, 0],
            dets[:, 1],
            dets[:, 2],
            dets[:, 3],
            dets[:, 4],
        )
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep

    def predict(
        self, img: np.ndarray, bboxes: List[Tuple[float, float, float, float]] = None
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """
        Wrapper around detect(), ignoring the `bboxes` argument.

        Args:
            img: HxWx3 BGR image.
            bboxes: Ignored for detection.

        Returns:
            dets: List of [x1, y1, x2, y2, score].
            kpss: List of keypoints (or None).
        """
        dets, kpss = self.detect(img)
        return dets.tolist(), (kpss.tolist() if kpss is not None else None)


class AgeGenderModel(FaceModelBase):
    """
    ONNX-based Age and Gender prediction model.

    Usage example:
        age_model = AgeGenderModel(model_path="genderage.onnx")
        age_model.prepare()
        results = age_model.predict(image, dets)
    """

    def __init__(
        self,
        model_path: str,
        det_size: Tuple[int, int] = (640, 640),
        face_size: Tuple[int, int] = (96, 96),
    ):
        assert model_path is not None and os.path.exists(
            model_path
        ), f"Model file not found: {model_path}"
        self.model_path: str = model_path

        # For this model, set mean and std appropriately
        self.input_mean: float = 0.0
        self.input_std: float = 1.0

        self.det_size: Tuple[int, int] = det_size
        self.face_size: Tuple[int, int] = face_size

        # Load and convert the ONNX model to PyTorch
        model = onnx.load(model_path)
        self.torch_model = convert(model)
        self.torch_model.eval()

    def prepare(self, *args, **kwargs):
        """
        Placeholder for any dynamic preparation of the age/gender model,
        such as adjusting normalization values.
        """
        pass  # No prep steps needed by default

    def preprocess_face(self, img: np.ndarray) -> torch.Tensor:
        """
        Crop/resize, normalize, and convert a face patch to a PyTorch tensor.

        Args:
            img: HxWx3 BGR or RGB face patch.

        Returns:
            A torch.Tensor of shape (3, H, W).
        """
        # Resize to fixed face_size (width, height)
        img_resized = cv2.resize(img, self.face_size)

        # Convert BGR→RGB if needed (assuming input may be BGR)
        img_rgb = img_resized[:, :, ::-1]

        # Convert to float32 and normalize
        img_float = img_rgb.astype(np.float32)
        img_norm = (img_float - self.input_mean) / self.input_std

        # HWC → CHW
        img_chw = img_norm.transpose(2, 0, 1)

        return torch.from_numpy(img_chw)

    def predict(
        self, img: np.ndarray, bboxes: List[Tuple[float, float, float, float]]
    ) -> List[Tuple[int, int]]:
        """
        Predict gender and age for each face in the given image.

        Args:
            img: Original image (HxWx3 BGR or RGB).
            bboxes: List of bounding boxes, each (x1, y1, x2, y2).

        Returns:
            List of (gender, age) tuples for each bounding box.
            - gender: 0 or 1 (e.g., 0=male, 1=female).
            - age: Predicted age in years.
        """
        results = []
        rotate = 0
        for bbox in bboxes:
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            scale = self.face_size[0] / (max(w, h) * 1.5)

            aligned_face, _ = transform(img, center, self.face_size[0], scale, rotate)

            if aligned_face is None or aligned_face.size == 0:
                results.append((None, None))
                continue

            tensor = self.preprocess_face(aligned_face).unsqueeze(0)
            with torch.no_grad():
                output = self.torch_model(tensor)[0].cpu().numpy()

            gender = int(np.argmax(output[:2]))
            age = int(np.round(output[2] * 100))
            results.append((gender, age))

        return results


class ActivationHook:
    """
    Utility class to register forward hooks on ReLU and Sigmoid layers
    in a PyTorch model and collect their activations.
    """

    def __init__(self):
        # Dictionary mapping layer names to a list of activation tensors
        self.features: defaultdict = defaultdict(list)

    def register_hooks(self, model: torch.nn.Module) -> List:
        """
        Attach hooks to every ReLU or Sigmoid module in `model`.

        Args:
            model: A torch.nn.Module instance.

        Returns:
            A list of handle objects for the registered hooks (to remove later).
        """
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.ReLU, nn.Sigmoid)):
                handle = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self.features[name].append(
                        out.detach().cpu()
                    )
                )
                handles.append(handle)
        return handles

    def get_activations(self) -> dict:
        """
        Return a dictionary mapping layer names to the first recorded activation.

        Returns:
            A dict where keys are layer names and values are NumPy arrays of activations.
        """
        single = {}
        for name, taps in self.features.items():
            if not taps:
                continue
            # Take the first activation recorded for this layer
            tensor = taps[0]
            single[name] = tensor.cpu().numpy()
        return single

    def clear(self):
        """
        Clear all stored activations.
        """
        self.features.clear()


def run_sanity_check(app, sample_image_path: str):
    """
    Sanity check: compare 'app.get(image)' against our RetinaFaceTorch + AgeGenderModel pipeline.
    """
    # 1) Load image
    image = cv2.imread(sample_image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {sample_image_path}")

    # 2) Get the "ground truth" or reference from app.get(...)
    actual = app.get(image)
    if not actual:
        print("app.get(image) returned no detections.")
    else:
        print("Actual predictions (from app.get):")
        # Assuming 'actual' is a list of objects with attributes .gender, .age, .bbox
        first = actual[0]
        print(f"  Gender: {first.gender}, Age: {first.age}")
        print(f"  BBox: {first.bbox}")

    # 3) Run our RetinaFaceTorch detector
    det_model_path = app.models["detection"].model_file
    rft = RetinaFaceTorch(model_file=det_model_path)
    rft.prepare(nms_thresh=0.4, det_thresh=0.5, input_size=(640, 640))

    dets, kpss = rft.predict(image)
    print("\nOur RetinaFaceTorch detections (x1, y1, x2, y2, score):")
    if dets:
        for det in dets:
            print(f"  {det}")
    else:
        print("  No detections.")

    # 4) Run our AgeGenderModel on those detections
    age_model_path = app.models["genderage"].model_file
    age_model = AgeGenderModel(model_path=age_model_path)
    age_model.prepare()

    # dets is a list of lists; convert to list of tuples
    det_tuples = [tuple(box[:4]) for box in dets]
    preds = age_model.predict(image, det_tuples)

    print("\nOur Age/Gender predictions (gender, age) for each detected box:")
    if preds:
        for (gender, age), bbox in zip(preds, dets):
            print(f"  BBox={tuple(bbox[:4])} → Gender={gender}, Age={age}")
    else:
        print("  No age/gender predictions (no detections).")


def extract_activations(
    subj: int,
    images_dir: str,
    activations_dir: str,
    retina_path: str,
    genderage_path: str,
):
    """
    Loop through all images for a given subject, run detection and age/gender
    models, collect activations via forward hooks, and save activations to disk.
    """
    # Read the list of image IDs from an Excel sheet
    faces_df = pd.read_excel(f"data/labels/subj_{subj:02d}/faces/faces_final.xlsx")
    cocoIds = (
        faces_df["cocoId"].tolist()
        + pd.read_excel(f"data/labels/shared/faces/faces_final.xlsx")["cocoId"].tolist()
    )

    # Initialize models
    rft = RetinaFaceTorch(model_file=retina_path)
    rft.prepare(nms_thresh=0.4, det_thresh=0.5)
    age_model = AgeGenderModel(model_path=genderage_path)

    # Create hooks
    retina_hook = ActivationHook()
    age_hook = ActivationHook()

    # Determine which layers to hook for each model
    retina_layers_to_hook = [
        name
        for name, mod in rft.torch_model.named_modules()
        if isinstance(mod, (nn.ReLU, nn.Sigmoid))
    ]
    age_layers_to_hook = [
        name
        for name, mod in age_model.torch_model.named_modules()
        if isinstance(mod, (nn.ReLU, nn.Sigmoid))
    ]

    # Register hooks
    retina_handles = retina_hook.register_hooks(rft.torch_model)
    age_handles = age_hook.register_hooks(age_model.torch_model)

    # Process each image
    for cocoId in tqdm(cocoIds, desc=f"Subject {subj} Activations"):
        img_name = f"{cocoId:012d}.jpg"
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Missing image: {img_path}")
            continue

        # Create output directories
        out_dir_detection = os.path.join(
            activations_dir, "detection", f"subj_{subj:02d}"
        )
        out_dir_genderage = os.path.join(
            activations_dir, "genderage", f"subj_{subj:02d}"
        )
        os.makedirs(out_dir_detection, exist_ok=True)
        os.makedirs(out_dir_genderage, exist_ok=True)

        # Detection activations
        det_npz_path = os.path.join(out_dir_detection, f"{cocoId:012d}.npz")
        if not os.path.exists(det_npz_path):
            dets, _ = rft.detect(img)  # Run detection (hooks fire here)
            _ = rft.forward(img)  # Additional forward pass to trigger hooks
            det_acts = retina_hook.get_activations()
            # Save activations as a .npz file
            np.savez(det_npz_path, **det_acts)
            retina_hook.clear()

        # Age/Gender activations
        age_npz_path = os.path.join(out_dir_genderage, f"{cocoId:012d}.npz")
        if not os.path.exists(age_npz_path):
            # Run age/gender prediction (hooks fire here)
            _ = age_model.predict(img, dets.tolist())
            age_acts = age_hook.get_activations()
            # Save activations as a .npz file
            np.savez(age_npz_path, **age_acts)
            age_hook.clear()

    # Remove all hooks to free resources
    for h in retina_handles + age_handles:
        h.remove()


def visualize_model_graph(
    retina_model: RetinaFaceTorch,
    age_model: AgeGenderModel,
):
    """
    Generate and save a PDF visualization of the RetinaFaceTorch model architecture using torchviz.
    """
    # Create a dummy input matching the expected input shape: (batch=1, channels=3, height=640, width=640)
    dummy_input = torch.randn(1, 3, 640, 640)
    retina_model.torch_model.eval()
    output = retina_model.torch_model(dummy_input)

    # Build the graph including model parameters
    dot = make_dot(output[0], params=dict(retina_model.torch_model.named_parameters()))
    # Save the graph to a PDF file
    dot.render(
        os.path.join("data", "neural_net_results", "retinaface_graph"), format="pdf"
    )

    # dummy_input = torch.randn(1, 3, 96, 96)
    # age_model.torch_model.eval()
    # output = age_model.torch_model(dummy_input)

    # # Build the graph including model parameters
    # dot = make_dot(output, params=dict(age_model.torch_model.named_parameters()))
    # print(dot)
    # # Save the graph to a PDF file
    # dot.render(os.path.join("data", "neural_net_results", "genderage_graph"), format="pdf")
    # 1) Do a forward pass
    dummy_input = torch.randn(1, 3, 96, 96)
    age_model.torch_model.eval()
    output = age_model.torch_model(dummy_input)

    # 2) Build the full dot graph (with every grad/param node)
    dot = make_dot(output, params=dict(age_model.torch_model.named_parameters()))

    # 3) Filter out:
    #    • Any node whose label is "AccumulateGrad" (autograd)
    #    • Any node whose label is "TBackward"    (autograd)
    #    • Any parameter whose name ends in ".bias"
    filtered_body = []
    for line in dot.body:
        if "AccumulateGrad" in line or "TBackward" in line or ".bias" in line:
            continue
        filtered_body.append(line)

    # 4) Overwrite dot.body and render
    dot.body = filtered_body
    dot.render("simplified_graph", format="pdf", cleanup=True)


def _numeric_key_detection(name: str) -> int:
    return int(name.split("_")[-1])


def _numeric_key_genderage(name: str) -> int:
    return int(name.split("_")[1])


from multiprocessing import Process, Queue

from rsa.permutation_analysis import run_mantel_test


def _process_layer(
    module: str,
    layer: str,
    subj_dir: str,
    cocoIds: list,
    metadata: list,
    subj: int,
    B: int,
    output_queue: Queue,
    bar: tqdm,
):
    """
    Worker function for a single layer. Re-loads activations for `layer` from disk,
    computes X, RDM, MDS, runs Mantel tests for all feature selections, and
    puts the list of result‐dicts into `output_queue`.
    """
    # try:
    feature_selections = ["centers", "sizes", "ages", "genders"]

    # 1) Gather activations for this layer across images
    features = []
    valid_ids = []
    for cocoId in cocoIds:
        file_path = os.path.join(subj_dir, f"{cocoId:012d}.npz")
        if not os.path.exists(file_path):
            continue
        data = np.load(file_path)
        if layer not in data:
            continue
        feat = data[layer].squeeze().flatten()
        features.append(feat)
        valid_ids.append(cocoId)

    results = []
    if len(features) < 2:
        # Nothing to do, return empty list
        output_queue.put(results)
        return

    # 2) Stack into a matrix: n_images × n_features
    X = np.vstack(features)

    bar.set_postfix_str(f"'{layer}': {X.shape}")

    # 3) Compute Representational Dissimilarity Matrix (RDM)
    D_X = pdist(X, metric="correlation")
    rdm = squareform(D_X)

    # 4) Apply Multidimensional Scaling (MDS) once per layer
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    X_mds = mds.fit_transform(rdm)

    # 5) For each feature_selection, run Mantel test using X_mds
    for feature_selection in feature_selections:
        mantel_out = run_mantel_test(
            D_X, X_mds, subj, metadata, B=B, feature_selection=feature_selection
        )
        #         return {
        #     "MDS": {
        #         "subject": subject_i,
        #         "r_obs": result_MDS["r_obs"],
        #         "p_value": result_MDS["p_value"],
        #         "feature": feature_selection,
        #     },
        #     "RDM": {
        #         "subject": subject_i,
        #         "r_obs": result_RDM["r_obs"],
        #         "p_value": result_RDM["p_value"],
        #         "feature": feature_selection,
        #     },
        # }

        for distance_space in mantel_out:
            results.append(
                {
                    "subject": mantel_out[distance_space].get("subject", subj),
                    "module": module,
                    "layer": layer,
                    "n_images": len(valid_ids),
                    "r_obs": round(mantel_out[distance_space]["r_obs"], 3),
                    "p_value": round(mantel_out[distance_space]["p_value"], 3),
                    "shape": X.shape,
                    "B": B,
                    "feature_selection": feature_selection,
                    "distance_space": distance_space,
                }
            )

    # 6) Return results for this layer
    output_queue.put(results)

    # except Exception as e:
    #     print(e)
    #     print(mantel_out)
    #     # If anything fails in the child, still put an empty list
    #     results.append({
    #         "subject": mantel_out.get("subject", subj),
    #         "module": module,
    #         "layer": layer,
    #         "n_images": len(valid_ids),
    #         "r_obs": np.nan,
    #         "p_value": np.nan,
    #         "shape":X.shape,
    #         "B": B,
    #         "feature_selection": feature_selection,
    #         "distance_space" : "ERROR"
    #     })
    #     output_queue.put(results)
    #     raise  # so that the exitcode != 0 is evident to the parent


def run_mantel_test_analysis(
    subj: int,
    activations_dir: str,
    output_excel: str,
    metadata: List[str],
    modules: List[str] = None,
    B: int = 2000,
):
    """
    Load saved activations for specified modules, compute Representational Dissimilarity Matrices (RDMs),
    run MDS once per layer, perform Mantel tests for all four feature selections (two per module),
    and save results to an Excel file—creating its directory if needed and saving incrementally
    after each layer to avoid data loss on error.

    Args:
        subj: Subject ID (integer).
        activations_dir: Base directory where activations are stored.
        output_excel: Path to the output Excel file.
        metadata: List of image identifiers (strings) for Mantel test.
        modules: List of modules to analyze (e.g., ["detection", "genderage"]).
                 Defaults to ["detection", "genderage"] if None.
        B: Number of permutations for Mantel test.
    """

    # If no modules specified, analyze both modules by default (four features total)
    if modules is None:
        modules = ["detection", "genderage"]

    results = []

    # Ensure the directory for output_excel exists
    output_dir = os.path.dirname(output_excel) or "."
    os.makedirs(output_dir, exist_ok=True)

    # If results already exist, skip computation
    if os.path.exists(output_excel):
        print(
            f"Output file {output_excel} already exists; skipping Mantel test analysis."
        )
        return

    # Load the list of image IDs from the corresponding Excel file
    faces_df = pd.read_excel(f"data/labels/subj_{subj:02d}/faces/faces_final.xlsx")
    cocoIds = (
        faces_df["cocoId"].tolist()
        + pd.read_excel(f"data/labels/shared/faces/faces_final.xlsx")["cocoId"].tolist()
    )

    for module in modules:
        print(f"=== Processing module: {module} ===")
        subj_dir = os.path.join(activations_dir, module, f"subj_{subj:02d}")
        npz_files = glob.glob(os.path.join(subj_dir, "*.npz"))
        if not npz_files:
            print(f"No .npz files found in {subj_dir}; skipping.")
            continue

        # Collect the union of all layer names across all .npz files
        all_layers = set()
        for path in npz_files:
            data = np.load(path)
            all_layers.update(data.files)

        # For simplicity, you can restrict to a subset of layers by uncommenting below.
        # But by default we'll use all layers.
        # if module == "detection":
        #     all_layers = ['Relu_30']
        # elif module == "genderage":
        #     all_layers = [
        #         'conv_8_relu', 'conv_9_dw_relu', 'conv_9_relu', 'conv_10_dw_relu',
        #         'conv_10_relu', 'conv_11_dw_relu', 'conv_11_relu', 'conv_12_dw_relu',
        #         'conv_12_relu', 'conv_13_dw_t0_relu', 'conv_13_t0_relu', 'conv_14_dw_t0_relu',
        #         'conv_14_t0_relu', 'conv_13_dw_t1_relu', 'conv_13_t1_relu', 'conv_14_dw_t1_relu'
        #     ]

        feature_selections = ["centers", "sizes", "ages", "genders"]

        # Process each layer
        # for layer in tqdm(sorted(all_layers), desc=f"{module} layers"):
        # bar = tqdm(total=len(all_layers), desc=f"{module} layers")

        if module == "detection":
            sorted_layers = sorted(all_layers, key=_numeric_key_detection)
            print(f"Found {len(sorted_layers)} layers: {sorted_layers}")

        else:
            sorted_layers = sorted(all_layers, key=_numeric_key_genderage)
            print(f"Found {len(sorted_layers)} layers: {sorted_layers}")

        bar = tqdm(range(len(all_layers)), desc=f"{module} layers")
        for i in bar:

            layer = sorted_layers[i]
            # Spawn a child process to compute RDM, MDS, and Mantel tests
            queue = Queue()
            p = Process(
                target=_process_layer,
                args=(module, layer, subj_dir, cocoIds, metadata, subj, B, queue, bar),
            )
            p.start()
            p.join()

            if p.exitcode == 0:
                # Child succeeded; retrieve its results
                try:
                    layer_results = queue.get_nowait()
                except Exception:
                    layer_results = []
                if layer_results:
                    results.extend(layer_results)

                # print("success")
                # print(results)
                # print(layer_results)

                # Save after each layer finishes
                df_partial = pd.DataFrame(results)
                df_partial.to_excel(output_excel, index=False)
            else:
                # Nonzero exitcode means child crashed or was killed
                print(
                    f"  [layer={layer}] child process exited with code {p.exitcode}, skipping."
                )

        print()  # blank line between modules

    # 8) Once all modules/layers processed, print final results
    df_results = pd.DataFrame(results)
    print("Final Mantel test results:")
    print(df_results)
    print(f"Saved Mantel test results to {output_excel}")


def main():
    """
    Main execution function. Adjust the file paths and subject IDs as needed.
    """
    # Paths and parameters (customize these as needed)
    subj = 1
    config = load_config("config.yaml")
    images_dir = config.directories.images_target_dir
    activations_dir = config.directories.activations_dir

    print("\n=== Initializing Face Analysis ===")
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "genderage"],
    )
    app.prepare(ctx_id=0)

    retina_model_path = app.models["detection"].model_file  # ← Update this path
    age_gender_model_path = app.models["genderage"].model_file  # ← Update this path
    sample_image_path = os.path.join(
        config.directories.images_target_dir, "000000531961.jpg"
    )  # ← Update if needed

    print("\n=== Running Sanity Check ===")
    run_sanity_check(app, sample_image_path)
    # Output files
    mantel_output_excel = os.path.join(
        "data", "neural_net_results", f"subj_{subj:02d}", f"results.xlsx"
    )

    # 1) Example usage: detect faces, predict age/gender

    # 2) Extract and save activations for all images of subject `subj`
    print("\n=== Extracting and Saving Activations ===")
    extract_activations(
        subj=subj,
        images_dir=images_dir,
        activations_dir=activations_dir,
        retina_path=retina_model_path,
        genderage_path=age_gender_model_path,
    )

    # 3) Visualize the RetinaFaceTorch model graph and save as PDF
    print("\n=== Visualizing Model Graph ===")
    retina_model = RetinaFaceTorch(model_file=retina_model_path)
    age_model = AgeGenderModel(model_path=age_gender_model_path)
    # visualize_model_graph(retina_model, age_model)

    # 4) Run Mantel test analysis on saved activations
    print("\n=== Running Mantel Test Analysis ===")
    # Build metadata list of string IDs for Mantel test (must match order of images)
    faces_df = pd.read_excel(f"data/labels/subj_{subj:02d}/faces/faces_final.xlsx")
    face_df_shared = pd.read_excel(f"data/labels/shared/faces/faces_final.xlsx")

    cocoIds = faces_df["cocoId"].tolist() + face_df_shared["cocoId"].tolist()
    metadata = [f"{x:012d}" for x in cocoIds]

    run_mantel_test_analysis(
        subj=subj,
        activations_dir=activations_dir,
        output_excel=mantel_output_excel,
        metadata=metadata,
        modules=[
            "detection",
            "genderage",
        ],  # Change to ["detection", "genderage"] if needed
        B=2000,
    )


if __name__ == "__main__":
    main()
