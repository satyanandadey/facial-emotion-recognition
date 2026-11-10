"""
Decision-level facial emotion fusion: Appearance CNN + dlib Geometry Random Forest.

Expected project layout:
MiniProject/
    assets/shape_predictor_68_face_landmarks.dat
    models/appearance_cnn_fixed.keras  (or appearance_cnn.keras)
    models/dlib_geometry_model_expanded.joblib
    results/fusion_config.json
    fusion_model.py

CNN preprocessing must match the training pipeline. The current implementation
uses grayscale -> 48x48 -> per-image min-max normalization to [0, 255]
(float32), matching the preprocessing currently documented in the fusion
notebook. If the original CNN training used another transformation, update
_preprocess_cnn_image() here and preprocess_fer_image() in the notebook together.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import dlib
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


CNN_CLASSES = [
    "angry", "disgust", "fear", "happy",
    "sad", "surprise", "neutral"
]


def normalize_landmarks(points: np.ndarray) -> Optional[np.ndarray]:
    """Normalize 68 landmarks by eye midpoint, rotation, and interocular distance."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        return None

    left_eye_center = points[36:42].mean(axis=0)
    right_eye_center = points[42:48].mean(axis=0)
    eyes_midpoint = (left_eye_center + right_eye_center) / 2.0

    translated_points = points - eyes_midpoint
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.arctan2(dy, dx)

    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_points = translated_points @ rotation.T

    left_rotated = (left_eye_center - eyes_midpoint) @ rotation.T
    right_rotated = (right_eye_center - eyes_midpoint) @ rotation.T
    interocular_distance = np.linalg.norm(right_rotated - left_rotated)

    if not np.isfinite(interocular_distance) or interocular_distance < 1e-6:
        return None

    return rotated_points / interocular_distance


def calculate_geometric_features(shape: Any) -> Optional[Dict[str, float]]:
    """Extract the same 25 engineered features used by Geometry_Branch.ipynb."""
    points = np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(68)],
        dtype=np.float64
    )
    norm_points = normalize_landmarks(points)
    if norm_points is None:
        return None

    def dist(p1: int, p2: int) -> float:
        return float(np.linalg.norm(norm_points[p1] - norm_points[p2]))

    def aspect_ratio(indices) -> float:
        vertical_1 = dist(indices[1], indices[5])
        vertical_2 = dist(indices[2], indices[4])
        horizontal = dist(indices[0], indices[3])
        return (
            (vertical_1 + vertical_2) / (2.0 * horizontal)
            if horizontal > 0 else 0.0
        )

    features: Dict[str, float] = {}
    features["left_eye_ar"] = aspect_ratio([36, 37, 38, 39, 40, 41])
    features["right_eye_ar"] = aspect_ratio([42, 43, 44, 45, 46, 47])
    features["eye_ar_asymmetry"] = abs(
        features["left_eye_ar"] - features["right_eye_ar"]
    )

    features["left_inner_brow_eye"] = dist(21, 39)
    features["right_inner_brow_eye"] = dist(22, 42)
    features["left_outer_brow_eye"] = dist(17, 36)
    features["right_outer_brow_eye"] = dist(26, 45)
    features["left_brow_slope"] = float(norm_points[21][1] - norm_points[17][1])
    features["right_brow_slope"] = float(norm_points[22][1] - norm_points[26][1])

    features["nose_width"] = dist(31, 35)
    features["nose_bridge_length"] = dist(27, 30)
    features["nose_mouth_dist"] = dist(33, 51)

    features["mouth_width"] = dist(48, 54)
    features["mouth_opening"] = dist(51, 57)
    features["mouth_ar"] = (
        features["mouth_opening"] / features["mouth_width"]
        if features["mouth_width"] > 0 else 0.0
    )
    features["upper_lip_thickness"] = dist(51, 62)
    features["lower_lip_thickness"] = dist(57, 66)

    mouth_center = norm_points[48:60].mean(axis=0)
    features["left_lip_corner_y"] = float(norm_points[48][1] - mouth_center[1])
    features["right_lip_corner_y"] = float(norm_points[54][1] - mouth_center[1])

    features["jaw_width"] = dist(3, 13)
    features["lower_jaw_width"] = dist(5, 11)
    features["chin_mouth_dist"] = dist(8, 57)
    features["lower_face_height"] = dist(33, 8)
    features["mouth_asymmetry"] = abs(dist(48, 51) - dist(54, 51))
    features["jaw_asymmetry"] = abs(dist(3, 8) - dist(13, 8))

    if len(features) != 25 or not np.isfinite(list(features.values())).all():
        return None
    return features


class FacialEmotionFusion:
    """CNN + dlib/Random-Forest probability fusion with CNN-only fallback."""

    def __init__(self, project_dir: Union[str, Path] = None):
        self.project_dir = (
            Path(project_dir).resolve()
            if project_dir is not None
            else Path(__file__).resolve().parent
        )
        self.model_dir = self.project_dir / "models"
        self.assets_dir = self.project_dir / "assets"
        self.results_dir = self.project_dir / "results"

        cnn_candidates = [
            self.model_dir / "appearance_cnn_fixed.keras",
            self.model_dir / "appearance_cnn.keras",
        ]
        cnn_path = next((p for p in cnn_candidates if p.is_file()), None)

        geometry_path = self.model_dir / "dlib_geometry_model_expanded.joblib"
        predictor_path = self.assets_dir / "shape_predictor_68_face_landmarks.dat"
        config_path = self.results_dir / "fusion_config.json"

        required = [geometry_path, predictor_path, config_path]
        missing = [str(path) for path in required if not path.is_file()]
        if cnn_path is None:
            missing.append(
                str(self.model_dir / "appearance_cnn_fixed.keras")
                + " (or appearance_cnn.keras)"
            )
        if missing:
            raise FileNotFoundError(
                "Required project files are missing:\n" + "\n".join(missing)
            )

        # Inference only: no need to restore optimizer/loss compilation state.
        self.cnn_model = tf.keras.models.load_model(cnn_path, compile=False)

        geometry_artifact = joblib.load(geometry_path)
        self.geometry_model = geometry_artifact["model"]
        self.geometry_scaler = geometry_artifact["scaler"]
        self.feature_columns = list(geometry_artifact["feature_columns"])
        self.geometry_classes = [str(x) for x in geometry_artifact["emotion_classes"]]

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        # In this project, alpha is the CNN weight:
        # fused = (CNN weight * CNN probabilities)
        #       + (Geometry weight * Geometry probabilities).
        self.alpha = float(config["best_alpha"])
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("Fusion alpha must be between 0 and 1.")

        self.cnn_weight = float(config.get("cnn_weight", self.alpha))
        self.geometry_weight = float(
            config.get("geometry_weight", 1.0 - self.cnn_weight)
        )

        if (
            not 0.0 <= self.cnn_weight <= 1.0
            or not 0.0 <= self.geometry_weight <= 1.0
        ):
            raise ValueError("CNN and geometry weights must be between 0 and 1.")

        if not np.isclose(self.cnn_weight + self.geometry_weight, 1.0):
            raise ValueError(
                "CNN and geometry weights must add up to 1.0; "
                f"got {self.cnn_weight} + {self.geometry_weight}."
            )

        if not np.isclose(self.alpha, self.cnn_weight):
            raise ValueError(
                "fusion_config.json is inconsistent: best_alpha must equal "
                "cnn_weight because alpha represents the CNN weight."
            )

        self.cnn_classes = list(config.get("class_order", CNN_CLASSES))
        if self.cnn_classes != CNN_CLASSES:
            raise ValueError(
                f"Unexpected CNN class order in fusion_config.json: {self.cnn_classes}"
            )

        if len(self.feature_columns) != 25:
            raise ValueError(
                f"Expected 25 geometry features, found {len(self.feature_columns)}."
            )

        if set(self.geometry_classes) != set(self.cnn_classes):
            raise ValueError(
                "Geometry and CNN class labels do not match: "
                f"{self.geometry_classes} vs {self.cnn_classes}"
            )

        if tuple(self.cnn_model.input_shape[1:]) != (48, 48, 1):
            raise ValueError(
                f"Expected CNN input shape (48, 48, 1), got {self.cnn_model.input_shape}."
            )
        if int(self.cnn_model.output_shape[-1]) != len(self.cnn_classes):
            raise ValueError("CNN output count does not match configured emotion classes.")

        self.detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(str(predictor_path))

        self.cnn_model_path = cnn_path

    @staticmethod
    def _read_gray(image: Union[str, Path, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Could not read image: {image}")
            return gray

        array = np.asarray(image)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        elif array.ndim == 3 and array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif array.ndim != 2:
            raise ValueError("Input must be grayscale, BGR, or BGRA.")

        if np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("Image contains NaN or infinite pixel values.")
            if array.size and float(array.max()) <= 1.0:
                array = array * 255.0

        return np.clip(array, 0, 255).astype(np.uint8)

    @staticmethod
    def _preprocess_cnn_image(gray_image: np.ndarray) -> np.ndarray:
        """Must match preprocess_fer_image() in the fusion notebook."""
        resized = cv2.resize(
            gray_image, (48, 48), interpolation=cv2.INTER_AREA
        )
        normalized = cv2.normalize(
            resized, None, 0, 255, cv2.NORM_MINMAX
        )
        tensor = normalized.astype(np.float32)[..., np.newaxis]
        return tensor[np.newaxis, ...]

    def _find_face_and_shape(self, gray: np.ndarray):
        faces = self.detector(gray, 2)
        predictor_image = gray
        scale = 1.0

        if len(faces) == 0:
            predictor_image = cv2.resize(
                gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC
            )
            scale = 3.0
            faces = self.detector(predictor_image, 2)

        if len(faces) == 0:
            return None, None

        face = max(
            faces,
            key=lambda rect: max(0, rect.width()) * max(0, rect.height())
        )
        shape = self.landmark_predictor(predictor_image, face)

        x1 = max(0, int(face.left() / scale))
        y1 = max(0, int(face.top() / scale))
        x2 = min(gray.shape[1], int(face.right() / scale))
        y2 = min(gray.shape[0], int(face.bottom() / scale))

        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            crop = None

        return shape, crop

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        use_face_crop_for_cnn: bool = False
    ) -> Dict[str, Any]:
        """
        Predict one image.

        By default, the CNN receives the full image, matching the FER-2013
        evaluation pipeline. Set use_face_crop_for_cnn=True only for camera/
        scene images when a face crop is intentionally desired and consistent
        with the CNN's expected input distribution.
        """
        gray = self._read_gray(image)
        shape, face_crop = self._find_face_and_shape(gray)

        cnn_source = (
            face_crop
            if use_face_crop_for_cnn and face_crop is not None
            else gray
        )
        cnn_tensor = self._preprocess_cnn_image(cnn_source)
        cnn_probs = np.asarray(
            self.cnn_model.predict(cnn_tensor, verbose=0)[0],
            dtype=np.float64
        )

        if (
            cnn_probs.shape != (len(self.cnn_classes),)
            or not np.isfinite(cnn_probs).all()
            or np.any(cnn_probs < 0)
            or cnn_probs.sum() <= 0
        ):
            raise ValueError("CNN returned invalid class probabilities.")

        cnn_probs = cnn_probs / cnn_probs.sum()
        cnn_prob_dict = {
            emotion: float(cnn_probs[i])
            for i, emotion in enumerate(self.cnn_classes)
        }

        geometry_available = False
        geometry_prob_dict = None
        fused_probs = cnn_probs.copy()
        method = "cnn_fallback"

        if shape is not None:
            features = calculate_geometric_features(shape)

            if features is not None:
                feature_frame = pd.DataFrame(
                    [[features[name] for name in self.feature_columns]],
                    columns=self.feature_columns,
                    dtype=np.float64
                )
                scaled_features = self.geometry_scaler.transform(feature_frame)
                geometry_raw = self.geometry_model.predict_proba(scaled_features)[0]

                geometry_by_class = {
                    str(label): float(geometry_raw[i])
                    for i, label in enumerate(self.geometry_model.classes_)
                }
                geometry_probs = np.asarray(
                    [geometry_by_class[label] for label in self.cnn_classes],
                    dtype=np.float64
                )

                if (
                    np.isfinite(geometry_probs).all()
                    and np.all(geometry_probs >= 0)
                    and geometry_probs.sum() > 0
                ):
                    geometry_probs /= geometry_probs.sum()
                    geometry_prob_dict = {
                        emotion: float(geometry_probs[i])
                        for i, emotion in enumerate(self.cnn_classes)
                    }

                    fused_probs = (
                        self.cnn_weight * cnn_probs
                        + self.geometry_weight * geometry_probs
                    )
                    geometry_available = True
                    method = "cnn_geometry_fusion"

        fused_prob_dict = {
            emotion: float(fused_probs[i])
            for i, emotion in enumerate(self.cnn_classes)
        }
        predicted_emotion = max(fused_prob_dict, key=fused_prob_dict.get)

        return {
            "emotion": predicted_emotion,
            "confidence": fused_prob_dict[predicted_emotion],
            "method": method,
            "geometry_available": geometry_available,
            "alpha": self.alpha,  # Kept for compatibility; alpha is CNN weight.
            "cnn_weight": self.cnn_weight,
            "geometry_weight": self.geometry_weight,
            "probabilities": fused_prob_dict,
            "cnn_probabilities": cnn_prob_dict,
            "geometry_probabilities": geometry_prob_dict,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict facial emotion using CNN + dlib geometry fusion."
    )
    parser.add_argument("image", help="Path to an input image.")
    parser.add_argument(
        "--face-crop-cnn",
        action="store_true",
        help="Use the detected face crop as CNN input instead of the full image."
    )
    args = parser.parse_args()

    fusion_system = FacialEmotionFusion()
    prediction = fusion_system.predict(
        args.image,
        use_face_crop_for_cnn=args.face_crop_cnn
    )

    print("Predicted emotion:", prediction["emotion"])
    print("Confidence:", f"{prediction['confidence']:.2%}")
    print("Prediction method:", prediction["method"])
    print("Geometry available:", prediction["geometry_available"])
