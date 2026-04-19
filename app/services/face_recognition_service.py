from __future__ import annotations

import base64
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from flask import current_app

from ..extensions import db
from ..models import Student


@dataclass
class EnrollmentResult:
    photo_path: str
    dataset_path: str
    message: str
    sample_count: int


@dataclass
class TrainingResult:
    student_count: int
    sample_count: int
    trainer_path: str
    labels_path: str
    message: str


@dataclass
class RecognitionResult:
    status: str
    label: str
    student_id: str | None
    confidence_percent: float
    threshold_score: float
    remarks: str
    face_detected: bool
    capture_path: str = ""


@dataclass
class FaceSample:
    face: np.ndarray
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float
    aligned: bool


class FaceRecognitionService:
    """Professional OpenCV recognition with quality filtering and model consensus."""

    def __init__(self) -> None:
        self.face_size = int(current_app.config["FACE_IMAGE_SIZE"])
        self.threshold_score = float(current_app.config["FACE_MATCH_THRESHOLD"])
        self.distance_ceiling = float(current_app.config["FACE_DISTANCE_CEILING"])
        self.dataset_root = Path(current_app.config["DATASET_FOLDER"])
        self.capture_root = Path(current_app.config["CAPTURE_FOLDER"])
        self.trainer_file = Path(current_app.config["TRAINER_FILE"])
        self.labels_file = Path(current_app.config["LABELS_FILE"])
        self.model_files = {
            "lbph": self.trainer_file,
            "eigen": self.trainer_file.with_name("trainer_eigen.yml"),
            "fisher": self.trainer_file.with_name("trainer_fisher.yml"),
        }
        self.model_weights = {"lbph": 0.5, "fisher": 0.3, "eigen": 0.2}
        self.default_distance_ceilings = {
            "lbph": self.distance_ceiling,
            "eigen": 6000.0,
            "fisher": 1400.0,
        }
        self.default_acceptance_distances = {
            "lbph": max(self.distance_ceiling * self.threshold_score, 28.0),
            "eigen": 1800.0,
            "fisher": 320.0,
        }
        self.min_face_quality = float(current_app.config.get("FACE_MIN_QUALITY", 0.34))
        self.min_vote_ratio = float(current_app.config.get("FACE_MATCH_DOMINANCE", 0.55))
        self.min_consensus_frames = int(current_app.config.get("FACE_MIN_CONSENSUS_FRAMES", 2))
        self.max_enrollment_samples = int(current_app.config.get("FACE_MAX_ENROLLMENT_SAMPLES", 8))
        self._reference_cache: dict[str, dict[str, object]] | None = None

        cascade_root = Path(cv2.data.haarcascades)
        cascade_paths = [
            cascade_root / "haarcascade_frontalface_default.xml",
            cascade_root / "haarcascade_frontalface_alt.xml",
            cascade_root / "haarcascade_frontalface_alt2.xml",
        ]
        eye_paths = [
            cascade_root / "haarcascade_eye.xml",
            cascade_root / "haarcascade_eye_tree_eyeglasses.xml",
        ]
        self.cascades = [
            cv2.CascadeClassifier(str(path))
            for path in cascade_paths
            if path.exists()
        ]
        self.eye_cascades = [
            cv2.CascadeClassifier(str(path))
            for path in eye_paths
            if path.exists()
        ]

    @property
    def available(self) -> bool:
        return (
            hasattr(cv2, "face")
            and hasattr(cv2.face, "LBPHFaceRecognizer_create")
            and any(not cascade.empty() for cascade in self.cascades)
        )

    def enroll_student_face_from_camera(
        self,
        student: Student,
        camera_image_samples: list[str],
    ) -> EnrollmentResult:
        self._ensure_engine()
        detected_faces = self._extract_faces_from_samples(
            camera_image_samples,
            target_count=self.max_enrollment_samples,
            min_quality=self.min_face_quality,
        )
        if not detected_faces:
            raise ValueError("No face was detected. Keep the face centered and try again.")

        duplicate_match = self._match_existing_student(detected_faces, exclude_student_id=student.student_id)
        if duplicate_match is not None:
            raise ValueError(
                f"The captured face already matches {duplicate_match.label}. Duplicate enrollment was blocked."
            )

        dataset_dir = self.dataset_root / student.student_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for existing_file in dataset_dir.glob("sample_*.png"):
            existing_file.unlink(missing_ok=True)

        saved_paths: list[Path] = []
        for index, face_sample in enumerate(detected_faces, start=1):
            sample_path = dataset_dir / f"sample_{index:02d}.png"
            cv2.imwrite(str(sample_path), face_sample.face)
            saved_paths.append(sample_path)

        preview_path = saved_paths[0]
        student.photo_path = preview_path.as_posix()
        student.dataset_path = dataset_dir.as_posix()
        student.face_samples_count = len(saved_paths)
        db.session.flush()

        training_result = self.train_model()
        return EnrollmentResult(
            photo_path=preview_path.as_posix(),
            dataset_path=dataset_dir.as_posix(),
            message=(
                f"Captured {len(saved_paths)} face sample(s) for {student.name}. "
                f"Model refreshed with {training_result.student_count} enrolled student(s)."
            ),
            sample_count=len(saved_paths),
        )

    def train_model(self) -> TrainingResult:
        self._ensure_engine()
        students = Student.query.order_by(Student.student_id.asc()).all()
        training_faces: list[np.ndarray] = []
        labels: list[int] = []
        label_map: dict[str, dict[str, object]] = {}
        enrolled_students = 0

        for label_index, student in enumerate(students, start=1):
            sample_paths = self._student_sample_paths(student)
            if not sample_paths:
                student.face_model_label = None
                continue

            normalized_samples: list[np.ndarray] = []
            for sample_path in sample_paths:
                face_sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if face_sample is None:
                    continue
                normalized_samples.append(self._normalize_face(face_sample))

            if not normalized_samples:
                student.face_model_label = None
                continue

            student.face_model_label = label_index
            label_map[str(label_index)] = {
                "student_id": student.student_id,
                "name": student.name,
                "enrollment_no": student.enrollment_no,
            }
            enrolled_students += 1
            for normalized_face in normalized_samples:
                for variant in self._training_variants(normalized_face):
                    training_faces.append(variant)
                    labels.append(label_index)

        if not training_faces:
            for model_file in self.model_files.values():
                model_file.unlink(missing_ok=True)
            self.labels_file.unlink(missing_ok=True)
            self._reference_cache = None
            db.session.flush()
            raise ValueError("No student face dataset is available for training.")

        model_metadata = self._train_models(training_faces, labels, enrolled_students)
        self.labels_file.parent.mkdir(parents=True, exist_ok=True)
        self.labels_file.write_text(
            json.dumps(
                {
                    "labels": label_map,
                    "models": model_metadata,
                    "training": {
                        "student_count": enrolled_students,
                        "sample_count": len(training_faces),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._reference_cache = None
        db.session.flush()

        return TrainingResult(
            student_count=enrolled_students,
            sample_count=len(training_faces),
            trainer_path=self.model_files["lbph"].as_posix(),
            labels_path=self.labels_file.as_posix(),
            message=(
                f"Training completed with {enrolled_students} student(s) and "
                f"{len(training_faces)} processed face sample(s)."
            ),
        )

    def preview_student_match(self, student: Student, camera_image_samples: list[str]) -> RecognitionResult:
        self._ensure_engine()
        detected_faces = self._extract_faces_from_samples(
            camera_image_samples,
            target_count=3,
            min_quality=max(self.min_face_quality - 0.06, 0.24),
        )
        if not detected_faces:
            return RecognitionResult(
                status="NoFace",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="No face was detected from the webcam feed.",
                face_detected=False,
            )

        if student.face_samples_count == 0:
            return RecognitionResult(
                status="ReadyToEnroll",
                label=student.name,
                student_id=student.student_id,
                confidence_percent=0.0,
                threshold_score=0.0,
                remarks="Face detected. The system is ready to enroll this student automatically.",
                face_detected=True,
            )

        return self.verify_student_for_samples(student, camera_image_samples, prepared_faces=detected_faces)

    def verify_student_for_samples(
        self,
        student: Student,
        camera_image_samples: list[str],
        capture_group: str = "public_checkins",
        capture_identifier: str | None = None,
        prepared_faces: list[FaceSample] | None = None,
    ) -> RecognitionResult:
        self._ensure_engine()
        if student.face_samples_count == 0:
            return RecognitionResult(
                status="Unknown",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="The student has no enrolled face samples.",
                face_detected=False,
            )

        detected_faces = prepared_faces or self._extract_faces_from_samples(
            camera_image_samples,
            target_count=5,
            min_quality=max(self.min_face_quality - 0.04, 0.26),
        )
        capture_path = ""
        if camera_image_samples:
            capture_path = self._save_capture(
                camera_image_samples[0],
                capture_group,
                capture_identifier or student.student_id,
            ).as_posix()

        if not detected_faces:
            return RecognitionResult(
                status="NoFace",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="No face was detected from the webcam feed.",
                face_detected=False,
                capture_path=capture_path,
            )

        best_match = self._match_existing_student(detected_faces)
        if best_match is None:
            claimed_match = self._verify_claimed_student(student, detected_faces)
            if claimed_match is not None:
                claimed_match.capture_path = capture_path
                return claimed_match

            return RecognitionResult(
                status="Unknown",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="Unknown face detected. Attendance was not marked.",
                face_detected=True,
                capture_path=capture_path,
            )

        if best_match.student_id == student.student_id:
            return RecognitionResult(
                status="Present",
                label=best_match.label,
                student_id=best_match.student_id,
                confidence_percent=best_match.confidence_percent,
                threshold_score=best_match.threshold_score,
                remarks="Face recognized successfully.",
                face_detected=True,
                capture_path=capture_path,
            )

        return RecognitionResult(
            status="Mismatch",
            label=best_match.label,
            student_id=best_match.student_id,
            confidence_percent=best_match.confidence_percent,
            threshold_score=best_match.threshold_score,
            remarks=f"Captured face matches {best_match.label}, not the entered student.",
            face_detected=True,
            capture_path=capture_path,
        )

    def _verify_claimed_student(
        self,
        student: Student,
        detected_faces: list[FaceSample],
    ) -> RecognitionResult | None:
        models, label_map, model_stats = self._load_models()
        if not models or not label_map:
            return None

        references = self._build_reference_cache(label_map)
        student_reference = references.get(student.student_id)
        if not student_reference:
            return None

        verified_frames: list[dict[str, float]] = []
        for face_sample in detected_faces:
            probe_signature = self._build_face_signature(face_sample.face)
            target_similarity = self._reference_similarity(face_sample.face, probe_signature, student_reference)
            best_reference_id, best_reference_similarity, runner_up_similarity = self._best_reference_match(
                face_sample.face,
                probe_signature,
                references,
            )

            if target_similarity < 0.72:
                continue
            if best_reference_id and best_reference_id != student.student_id and best_reference_similarity > target_similarity + 0.04:
                continue
            if best_reference_id == student.student_id and target_similarity < runner_up_similarity + 0.015:
                target_similarity *= 0.94

            model_votes = 0
            model_weight_total = 0.0
            threshold_total = 0.0
            for model_name, recognizer in models.items():
                try:
                    predicted_label, raw_distance = recognizer.predict(face_sample.face)
                except cv2.error:
                    continue

                label_data = label_map.get(str(predicted_label))
                if not label_data or str(label_data.get("student_id")) != student.student_id:
                    continue

                threshold_score = self._distance_score(float(raw_distance), model_name, model_stats)
                if threshold_score > min(self.threshold_score + 0.24, 0.86):
                    continue

                weight = self.model_weights.get(model_name, 0.2)
                model_votes += 1
                model_weight_total += weight
                threshold_total += threshold_score * weight

            if model_votes == 0:
                continue

            average_threshold = threshold_total / max(model_weight_total, 0.0001)
            model_confidence = 1.0 - min(max(average_threshold, 0.0), 1.0)
            score = (target_similarity * 0.62) + (model_confidence * 0.30) + (face_sample.quality_score * 0.08)
            if score < 0.64:
                continue

            verified_frames.append(
                {
                    "score": min(score, 1.0),
                    "threshold_score": average_threshold,
                }
            )

        if not verified_frames:
            return None

        required_frames = 1 if len(detected_faces) <= 2 else 2
        if len(verified_frames) < required_frames:
            return None

        average_score = sum(frame["score"] for frame in verified_frames) / len(verified_frames)
        average_threshold = sum(frame["threshold_score"] for frame in verified_frames) / len(verified_frames)
        return RecognitionResult(
            status="Present",
            label=student.name,
            student_id=student.student_id,
            confidence_percent=round(min(average_score, 1.0) * 100.0, 2),
            threshold_score=round(min(max(average_threshold, 0.0), 1.0), 4),
            remarks=f"Face verified for entered enrollment number with {len(verified_frames)} confirmed frame(s).",
            face_detected=True,
        )

    def recognize_unknown_or_known(self, camera_image_samples: list[str]) -> RecognitionResult:
        self._ensure_engine()
        detected_faces = self._extract_faces_from_samples(
            camera_image_samples,
            target_count=5,
            min_quality=max(self.min_face_quality - 0.04, 0.26),
        )
        if not detected_faces:
            return RecognitionResult(
                status="NoFace",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="No face was detected from the webcam feed.",
                face_detected=False,
            )

        best_match = self._match_existing_student(detected_faces)
        if best_match is None:
            return RecognitionResult(
                status="Unknown",
                label="Unknown",
                student_id=None,
                confidence_percent=0.0,
                threshold_score=1.0,
                remarks="Unknown face detected.",
                face_detected=True,
            )
        return best_match

    def _ensure_engine(self) -> None:
        if not self.available:
            raise ValueError("OpenCV LBPH face-recognition support is not available on this machine.")

    def _train_models(
        self,
        training_faces: list[np.ndarray],
        labels: list[int],
        student_count: int,
    ) -> dict[str, dict[str, float | int]]:
        model_metadata: dict[str, dict[str, float | int]] = {}
        labels_array = np.array(labels, dtype=np.int32)

        for model_name in ("lbph", "fisher", "eigen"):
            recognizer = self._create_recognizer(model_name)
            model_file = self.model_files[model_name]
            model_file.unlink(missing_ok=True)

            if recognizer is None:
                continue
            if model_name == "fisher" and student_count < 2:
                continue

            try:
                recognizer.train(training_faces, labels_array)
                model_file.parent.mkdir(parents=True, exist_ok=True)
                recognizer.write(str(model_file))
                model_metadata[model_name] = self._calibrate_model(recognizer, training_faces, labels, model_name)
            except cv2.error:
                model_file.unlink(missing_ok=True)

        if "lbph" not in model_metadata:
            raise ValueError("The face-recognition model could not be trained on this machine.")

        return model_metadata

    def _calibrate_model(
        self,
        recognizer,
        training_faces: list[np.ndarray],
        labels: list[int],
        model_name: str,
    ) -> dict[str, float | int]:
        correct_distances: list[float] = []
        for face, label in zip(training_faces, labels):
            try:
                predicted_label, raw_distance = recognizer.predict(face)
            except cv2.error:
                continue
            if int(predicted_label) == int(label):
                correct_distances.append(float(raw_distance))

        default_acceptance = self.default_acceptance_distances.get(model_name, self.distance_ceiling * self.threshold_score)
        if correct_distances:
            percentile_95 = float(np.percentile(correct_distances, 95))
            mean_distance = float(np.mean(correct_distances))
            acceptance_distance = max(default_acceptance, percentile_95 * 2.1, mean_distance * 2.4)
        else:
            acceptance_distance = default_acceptance

        distance_ceiling = max(
            self.default_distance_ceilings.get(model_name, self.distance_ceiling),
            acceptance_distance / max(self.threshold_score, 0.01),
        )

        return {
            "acceptance_distance": round(acceptance_distance, 4),
            "distance_ceiling": round(distance_ceiling, 4),
            "calibration_samples": len(correct_distances),
        }

    def _match_existing_student(
        self,
        detected_faces: list[FaceSample],
        exclude_student_id: str | None = None,
    ) -> RecognitionResult | None:
        models, label_map, model_stats = self._load_models()
        if not models or not label_map:
            return None

        references = self._build_reference_cache(label_map)
        frame_matches: list[dict[str, float | str]] = []
        for face_sample in detected_faces:
            match = self._match_single_face(face_sample, models, label_map, model_stats, references, exclude_student_id)
            if match is not None:
                frame_matches.append(match)

        if not frame_matches:
            return None

        candidate_scores: dict[str, dict[str, float | str | int]] = {}
        for match in frame_matches:
            student_id = str(match["student_id"])
            bucket = candidate_scores.setdefault(
                student_id,
                {
                    "label": str(match["label"]),
                    "votes": 0,
                    "total_score": 0.0,
                    "threshold_total": 0.0,
                },
            )
            bucket["votes"] = int(bucket["votes"]) + 1
            bucket["total_score"] = float(bucket["total_score"]) + float(match["score"])
            bucket["threshold_total"] = float(bucket["threshold_total"]) + float(match["threshold_score"])

        ordered_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: (float(item[1]["total_score"]), int(item[1]["votes"])),
            reverse=True,
        )
        best_student_id, best_bucket = ordered_candidates[0]
        second_score = float(ordered_candidates[1][1]["total_score"]) if len(ordered_candidates) > 1 else 0.0
        required_votes = min(max(self.min_consensus_frames, 1), len(detected_faces))
        vote_ratio = int(best_bucket["votes"]) / max(len(detected_faces), 1)
        average_score = float(best_bucket["total_score"]) / max(int(best_bucket["votes"]), 1)
        average_threshold = float(best_bucket["threshold_total"]) / max(int(best_bucket["votes"]), 1)

        if int(best_bucket["votes"]) < required_votes and len(detected_faces) > 1:
            return None
        if vote_ratio < self.min_vote_ratio and (float(best_bucket["total_score"]) - second_score) < 0.18:
            return None
        if average_score < 0.56:
            return None

        return RecognitionResult(
            status="Known",
            label=str(best_bucket["label"]),
            student_id=best_student_id,
            confidence_percent=round(min(average_score, 1.0) * 100.0, 2),
            threshold_score=round(min(max(average_threshold, 0.0), 1.0), 4),
            remarks=f"Known face matched with {int(best_bucket['votes'])} verified frame(s).",
            face_detected=True,
        )

    def _match_single_face(
        self,
        face_sample: FaceSample,
        models: dict[str, object],
        label_map: dict[str, dict[str, object]],
        model_stats: dict[str, dict[str, float | int]],
        references: dict[str, dict[str, object]],
        exclude_student_id: str | None = None,
    ) -> dict[str, float | str] | None:
        candidate_support: dict[str, dict[str, float | str | int]] = {}

        for model_name, recognizer in models.items():
            try:
                predicted_label, raw_distance = recognizer.predict(face_sample.face)
            except cv2.error:
                continue

            label_data = label_map.get(str(predicted_label))
            if not label_data:
                continue

            student_id = str(label_data["student_id"])
            if exclude_student_id and student_id == exclude_student_id:
                continue

            threshold_score = self._distance_score(float(raw_distance), model_name, model_stats)
            if threshold_score > min(self.threshold_score + 0.16, 0.98):
                continue

            weight = self.model_weights.get(model_name, 0.2)
            support = weight * (1.0 - threshold_score)
            bucket = candidate_support.setdefault(
                student_id,
                {
                    "label": str(label_data["name"]),
                    "support": 0.0,
                    "weight_total": 0.0,
                    "threshold_total": 0.0,
                    "model_votes": 0,
                },
            )
            bucket["support"] = float(bucket["support"]) + support
            bucket["weight_total"] = float(bucket["weight_total"]) + weight
            bucket["threshold_total"] = float(bucket["threshold_total"]) + (threshold_score * weight)
            bucket["model_votes"] = int(bucket["model_votes"]) + 1

        if not candidate_support:
            return None

        ordered_support = sorted(
            candidate_support.items(),
            key=lambda item: float(item[1]["support"]),
            reverse=True,
        )
        best_student_id, best_bucket = ordered_support[0]
        total_support = sum(float(bucket["support"]) for _, bucket in ordered_support) or 1.0
        model_vote_share = float(best_bucket["support"]) / total_support
        model_agreement = int(best_bucket["model_votes"]) / max(len(models), 1)
        average_threshold = float(best_bucket["threshold_total"]) / max(float(best_bucket["weight_total"]), 0.0001)

        probe_signature = self._build_face_signature(face_sample.face)
        best_reference_id, best_reference_similarity, runner_up_similarity = self._best_reference_match(
            face_sample.face,
            probe_signature,
            references,
            exclude_student_id,
        )
        candidate_similarity = self._reference_similarity(
            face_sample.face,
            probe_signature,
            references.get(best_student_id),
        )

        if candidate_similarity < 0.38:
            return None
        if best_reference_id and best_reference_id != best_student_id and best_reference_similarity > candidate_similarity + 0.06:
            return None
        if candidate_similarity < runner_up_similarity + 0.03 and model_agreement < 0.6:
            return None

        model_confidence = 1.0 - min(max(average_threshold, 0.0), 1.0)
        consensus_factor = 0.48 + (0.22 * model_vote_share) + (0.20 * model_agreement) + (0.10 * face_sample.quality_score)
        final_score = ((model_confidence * 0.72) + (candidate_similarity * 0.28)) * consensus_factor
        if int(best_bucket["model_votes"]) == 1 and len(models) > 1:
            final_score *= 0.84
        if final_score < 0.50:
            return None

        return {
            "student_id": best_student_id,
            "label": str(best_bucket["label"]),
            "score": min(final_score, 1.0),
            "threshold_score": average_threshold,
        }

    def _load_models(
        self,
    ) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, dict[str, float | int]]]:
        if not self.labels_file.exists():
            return {}, {}, {}

        try:
            metadata = json.loads(self.labels_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}, {}, {}

        if isinstance(metadata, dict) and "labels" in metadata:
            label_map = metadata.get("labels", {})
            model_stats = metadata.get("models", {})
        else:
            label_map = metadata if isinstance(metadata, dict) else {}
            model_stats = {}

        models: dict[str, object] = {}
        for model_name, model_file in self.model_files.items():
            if not model_file.exists():
                continue
            recognizer = self._create_recognizer(model_name)
            if recognizer is None:
                continue
            try:
                recognizer.read(str(model_file))
            except cv2.error:
                continue
            models[model_name] = recognizer

        return models, label_map, model_stats if isinstance(model_stats, dict) else {}

    def _create_recognizer(self, model_name: str):
        if not hasattr(cv2, "face"):
            return None
        factory_map = {
            "lbph": getattr(cv2.face, "LBPHFaceRecognizer_create", None),
            "eigen": getattr(cv2.face, "EigenFaceRecognizer_create", None),
            "fisher": getattr(cv2.face, "FisherFaceRecognizer_create", None),
        }
        factory = factory_map.get(model_name)
        return factory() if callable(factory) else None

    def _extract_faces_from_samples(
        self,
        camera_image_samples: list[str],
        target_count: int,
        min_quality: float,
    ) -> list[FaceSample]:
        detected_faces: list[FaceSample] = []
        for camera_image_data in camera_image_samples:
            image = self._decode_data_url(camera_image_data)
            if image is None:
                continue
            face = self._extract_primary_face(image)
            if face is not None:
                detected_faces.append(face)
        if not detected_faces:
            return []
        return self._select_best_face_samples(detected_faces, target_count, min_quality)

    def _decode_data_url(self, data_url: str) -> np.ndarray | None:
        if not data_url:
            return None
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", data_url)
        if not match:
            raise ValueError("Camera image data is invalid.")

        image_bytes = base64.b64decode(match.group(2))
        buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    def _extract_primary_face(self, image: np.ndarray | None) -> FaceSample | None:
        if image is None:
            return None
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prepared_images = self._prepared_detection_images(grayscale)
        best_box = self._detect_best_face_box(prepared_images)
        if best_box is None:
            fallback_face = self._extract_center_fallback_face(grayscale)
            if fallback_face is None:
                return None
            aligned_face, aligned = self._align_face(fallback_face)
            return self._build_face_sample(aligned_face, grayscale.shape, None, aligned)

        x, y, w, h = best_box
        cropped = self._crop_face_region(grayscale, x, y, w, h)
        if cropped is None:
            return None
        aligned_face, aligned = self._align_face(cropped)
        return self._build_face_sample(aligned_face, grayscale.shape, best_box, aligned)

    def _build_face_sample(
        self,
        face_region: np.ndarray,
        image_shape: tuple[int, ...],
        face_box: tuple[int, int, int, int] | None,
        aligned: bool,
    ) -> FaceSample:
        quality_score, sharpness, brightness, contrast = self._score_face_quality(face_region, image_shape, face_box, aligned)
        normalized_face = self._normalize_face(face_region)
        return FaceSample(
            face=normalized_face,
            quality_score=quality_score,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            aligned=aligned,
        )

    def _prepared_detection_images(self, grayscale: np.ndarray) -> list[np.ndarray]:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grayscale)
        equalized = cv2.equalizeHist(grayscale)
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        return [grayscale, equalized, clahe, blurred]

    def _detect_best_face_box(self, prepared_images: list[np.ndarray]) -> tuple[int, int, int, int] | None:
        best_box: tuple[int, int, int, int] | None = None
        best_score = -1.0

        for prepared in prepared_images:
            image_height, image_width = prepared.shape[:2]
            min_face = max(36, int(min(image_width, image_height) * 0.12))
            search_configs = [
                {"scaleFactor": 1.05, "minNeighbors": 4, "minSize": (min_face, min_face)},
                {"scaleFactor": 1.08, "minNeighbors": 3, "minSize": (min_face, min_face)},
                {"scaleFactor": 1.12, "minNeighbors": 5, "minSize": (min_face + 8, min_face + 8)},
            ]

            for cascade in self.cascades:
                if cascade.empty():
                    continue
                for config in search_configs:
                    detections = cascade.detectMultiScale(prepared, **config)
                    for x, y, w, h in detections:
                        score = self._face_box_score(x, y, w, h, image_width, image_height)
                        if score > best_score:
                            best_score = score
                            best_box = (int(x), int(y), int(w), int(h))

        return best_box

    def _face_box_score(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        image_width: int,
        image_height: int,
    ) -> float:
        area_score = float(w * h)
        face_center_x = x + (w / 2.0)
        face_center_y = y + (h / 2.0)
        image_center_x = image_width / 2.0
        image_center_y = image_height / 2.0
        center_distance = ((face_center_x - image_center_x) ** 2 + (face_center_y - image_center_y) ** 2) ** 0.5
        max_distance = ((image_center_x) ** 2 + (image_center_y) ** 2) ** 0.5 or 1.0
        center_bonus = 1.0 - min(center_distance / max_distance, 1.0)
        return area_score * (0.65 + (center_bonus * 0.35))

    def _crop_face_region(
        self,
        grayscale: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray | None:
        margin = int(min(w, h) * 0.18)
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(grayscale.shape[1], x + w + margin)
        y1 = min(grayscale.shape[0], y + h + margin)
        cropped = grayscale[y0:y1, x0:x1]
        return cropped if cropped.size else None

    def _extract_center_fallback_face(self, grayscale: np.ndarray) -> np.ndarray | None:
        image_height, image_width = grayscale.shape[:2]
        if image_height < 120 or image_width < 120:
            return None

        crop_width = int(image_width * 0.56)
        crop_height = int(image_height * 0.72)
        x0 = max(0, (image_width - crop_width) // 2)
        y0 = max(0, int(image_height * 0.12))
        x1 = min(image_width, x0 + crop_width)
        y1 = min(image_height, y0 + crop_height)
        cropped = grayscale[y0:y1, x0:x1]
        if not cropped.size:
            return None
        if self._contains_eye_features(cropped):
            return cropped
        return None

    def _contains_eye_features(self, grayscale_face_region: np.ndarray) -> bool:
        if not self.eye_cascades:
            return False
        candidate = cv2.equalizeHist(grayscale_face_region)
        min_eye = max(10, int(min(candidate.shape[:2]) * 0.08))
        for cascade in self.eye_cascades:
            if cascade.empty():
                continue
            eyes = cascade.detectMultiScale(
                candidate,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_eye, min_eye),
            )
            if len(eyes) >= 1:
                return True
        return False

    def _align_face(self, grayscale_face_region: np.ndarray) -> tuple[np.ndarray, bool]:
        eye_pair = self._detect_eye_pair(grayscale_face_region)
        if eye_pair is None:
            return grayscale_face_region, False

        left_eye, right_eye = eye_pair
        left_center = (left_eye[0] + (left_eye[2] / 2.0), left_eye[1] + (left_eye[3] / 2.0))
        right_center = (right_eye[0] + (right_eye[2] / 2.0), right_eye[1] + (right_eye[3] / 2.0))
        angle = math.degrees(math.atan2(right_center[1] - left_center[1], right_center[0] - left_center[0]))
        rotation_center = ((left_center[0] + right_center[0]) / 2.0, (left_center[1] + right_center[1]) / 2.0)
        matrix = cv2.getRotationMatrix2D(rotation_center, angle, 1.0)
        rotated = cv2.warpAffine(
            grayscale_face_region,
            matrix,
            (grayscale_face_region.shape[1], grayscale_face_region.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated, True

    def _detect_eye_pair(
        self,
        grayscale_face_region: np.ndarray,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
        if not self.eye_cascades:
            return None

        search_height = max(int(grayscale_face_region.shape[0] * 0.62), 24)
        search_region = grayscale_face_region[:search_height, :]
        prepared = cv2.equalizeHist(search_region)
        min_eye = max(10, int(min(prepared.shape[:2]) * 0.08))

        detections: list[tuple[int, int, int, int]] = []
        for cascade in self.eye_cascades:
            if cascade.empty():
                continue
            eyes = cascade.detectMultiScale(
                prepared,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(min_eye, min_eye),
            )
            for x, y, w, h in eyes:
                detections.append((int(x), int(y), int(w), int(h)))

        if len(detections) < 2:
            return None

        best_pair: tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None = None
        best_score = -1.0

        for index, left_eye in enumerate(detections[:-1]):
            for right_eye in detections[index + 1 :]:
                eye_a, eye_b = sorted((left_eye, right_eye), key=lambda eye: eye[0])
                left_center_y = eye_a[1] + (eye_a[3] / 2.0)
                right_center_y = eye_b[1] + (eye_b[3] / 2.0)
                eye_distance = (eye_b[0] + (eye_b[2] / 2.0)) - (eye_a[0] + (eye_a[2] / 2.0))
                if eye_distance <= grayscale_face_region.shape[1] * 0.15:
                    continue
                if eye_distance >= grayscale_face_region.shape[1] * 0.7:
                    continue
                if abs(left_center_y - right_center_y) >= search_height * 0.18:
                    continue

                spacing_ratio = eye_distance / max(float(grayscale_face_region.shape[1]), 1.0)
                spacing_score = 1.0 - min(abs(spacing_ratio - 0.35) / 0.35, 1.0)
                alignment_score = 1.0 - min(abs(left_center_y - right_center_y) / max(search_height * 0.18, 1.0), 1.0)
                size_score = min(((eye_a[2] * eye_a[3]) + (eye_b[2] * eye_b[3])) / max(search_region.size * 0.02, 1.0), 1.0)
                pair_score = (spacing_score * 0.45) + (alignment_score * 0.35) + (size_score * 0.20)
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = (eye_a, eye_b)

        return best_pair

    def _score_face_quality(
        self,
        face_region: np.ndarray,
        image_shape: tuple[int, ...],
        face_box: tuple[int, int, int, int] | None,
        aligned: bool,
    ) -> tuple[float, float, float, float]:
        resized = cv2.resize(face_region, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
        sharpness_value = float(cv2.Laplacian(resized, cv2.CV_64F).var())
        brightness_value = float(np.mean(resized))
        contrast_value = float(np.std(resized))

        sharpness_score = min(sharpness_value / 160.0, 1.0)
        brightness_score = 1.0 - min(abs(brightness_value - 132.0) / 132.0, 1.0)
        contrast_score = min(contrast_value / 54.0, 1.0)

        if face_box is not None:
            image_height, image_width = image_shape[:2]
            area_ratio = (face_box[2] * face_box[3]) / max(float(image_width * image_height), 1.0)
            area_score = 1.0 - min(abs(area_ratio - 0.16) / 0.16, 1.0)
        else:
            area_score = 0.62 if aligned else 0.48

        alignment_score = 1.0 if aligned else 0.82
        quality_score = (
            (sharpness_score * 0.30)
            + (brightness_score * 0.18)
            + (contrast_score * 0.18)
            + (area_score * 0.18)
            + (alignment_score * 0.16)
        )
        return round(max(min(quality_score, 1.0), 0.0), 4), round(sharpness_score, 4), round(brightness_score, 4), round(contrast_score, 4)

    def _select_best_face_samples(
        self,
        detected_faces: list[FaceSample],
        target_count: int,
        min_quality: float,
    ) -> list[FaceSample]:
        ordered_faces = sorted(detected_faces, key=lambda sample: sample.quality_score, reverse=True)
        selected_faces: list[FaceSample] = []

        for face_sample in ordered_faces:
            if face_sample.quality_score < min_quality and selected_faces:
                continue
            if not self._is_diverse_face_sample(face_sample, selected_faces):
                continue
            selected_faces.append(face_sample)
            if len(selected_faces) >= target_count:
                break

        if not selected_faces and ordered_faces:
            selected_faces.append(ordered_faces[0])

        for face_sample in ordered_faces:
            if len(selected_faces) >= target_count:
                break
            if any(face_sample is selected_face for selected_face in selected_faces):
                continue
            if not self._is_diverse_face_sample(face_sample, selected_faces, strict=False):
                continue
            selected_faces.append(face_sample)

        return selected_faces[:target_count]

    def _is_diverse_face_sample(
        self,
        candidate: FaceSample,
        selected_faces: list[FaceSample],
        strict: bool = True,
    ) -> bool:
        if not selected_faces:
            return True

        for existing_face in selected_faces:
            correlation = self._pixel_similarity(candidate.face, existing_face.face)
            average_difference = float(np.mean(np.abs(candidate.face.astype(np.float32) - existing_face.face.astype(np.float32))))
            if strict and correlation > 0.992 and average_difference < 2.5:
                return False
            if not strict and correlation > 0.997 and average_difference < 1.2:
                return False
        return True

    def _normalize_face(self, grayscale_face: np.ndarray) -> np.ndarray:
        resized = cv2.resize(grayscale_face, (self.face_size, self.face_size), interpolation=cv2.INTER_LINEAR)
        denoised = cv2.bilateralFilter(resized, 5, 30, 30)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(denoised)
        return cv2.normalize(clahe, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def _training_variants(self, normalized_face: np.ndarray) -> list[np.ndarray]:
        clahe_variant = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(normalized_face)
        mirror_variant = cv2.flip(normalized_face, 1)
        mirror_clahe_variant = cv2.flip(clahe_variant, 1)
        sharpened_variant = cv2.addWeighted(normalized_face, 1.15, cv2.GaussianBlur(normalized_face, (0, 0), 1.0), -0.15, 0)
        return [normalized_face, clahe_variant, mirror_variant, mirror_clahe_variant, sharpened_variant]

    def _build_reference_cache(self, label_map: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
        if self._reference_cache is not None:
            return self._reference_cache

        references: dict[str, dict[str, object]] = {}
        for label_data in label_map.values():
            student_id = str(label_data.get("student_id", ""))
            if not student_id or student_id in references:
                continue

            faces: list[np.ndarray] = []
            signatures: list[np.ndarray] = []
            for sample_path in self._student_sample_paths_for_id(student_id):
                face_sample = cv2.imread(str(sample_path), cv2.IMREAD_GRAYSCALE)
                if face_sample is None:
                    continue
                normalized_face = self._normalize_face(face_sample)
                faces.append(normalized_face)
                signatures.append(self._build_face_signature(normalized_face))

            if faces:
                references[student_id] = {
                    "label": str(label_data.get("name", student_id)),
                    "faces": faces,
                    "signatures": signatures,
                }

        self._reference_cache = references
        return references

    def _reference_similarity(
        self,
        probe_face: np.ndarray,
        probe_signature: np.ndarray,
        reference_entry: dict[str, object] | None,
    ) -> float:
        if not reference_entry:
            return 0.0

        faces = reference_entry.get("faces", [])
        signatures = reference_entry.get("signatures", [])
        similarities: list[float] = []
        for reference_face, reference_signature in zip(faces, signatures):
            feature_similarity = self._cosine_similarity(probe_signature, reference_signature)
            pixel_similarity = self._pixel_similarity(probe_face, reference_face)
            similarities.append((feature_similarity * 0.65) + (pixel_similarity * 0.35))

        if not similarities:
            return 0.0

        best_scores = sorted(similarities, reverse=True)[:2]
        return round(sum(best_scores) / len(best_scores), 4)

    def _best_reference_match(
        self,
        probe_face: np.ndarray,
        probe_signature: np.ndarray,
        references: dict[str, dict[str, object]],
        exclude_student_id: str | None = None,
    ) -> tuple[str | None, float, float]:
        ranked_matches: list[tuple[str, float]] = []
        for student_id, reference_entry in references.items():
            if exclude_student_id and student_id == exclude_student_id:
                continue
            similarity = self._reference_similarity(probe_face, probe_signature, reference_entry)
            ranked_matches.append((student_id, similarity))

        if not ranked_matches:
            return None, 0.0, 0.0

        ranked_matches.sort(key=lambda item: item[1], reverse=True)
        best_student_id, best_similarity = ranked_matches[0]
        runner_up_similarity = ranked_matches[1][1] if len(ranked_matches) > 1 else 0.0
        return best_student_id, best_similarity, runner_up_similarity

    def _build_face_signature(self, face: np.ndarray) -> np.ndarray:
        lbp_histogram = self._lbp_histogram(face)
        intensity_histogram, _ = np.histogram(face.ravel(), bins=32, range=(0, 256))
        intensity_histogram = intensity_histogram.astype(np.float32)
        intensity_histogram /= max(float(intensity_histogram.sum()), 1.0)

        normalized_face = face.astype(np.float32) / 255.0
        gradient_x = cv2.Sobel(normalized_face, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(normalized_face, cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)
        gradient_histogram, _ = np.histogram(gradient_magnitude.ravel(), bins=16, range=(0.0, 1.5))
        gradient_histogram = gradient_histogram.astype(np.float32)
        gradient_histogram /= max(float(gradient_histogram.sum()), 1.0)

        signature = np.concatenate([lbp_histogram, intensity_histogram, gradient_histogram]).astype(np.float32)
        norm = np.linalg.norm(signature) or 1.0
        return signature / norm

    def _lbp_histogram(self, face: np.ndarray) -> np.ndarray:
        working_face = face.astype(np.uint8)
        center = working_face[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        lbp |= (np.uint8(working_face[:-2, :-2] >= center) << 7)
        lbp |= (np.uint8(working_face[:-2, 1:-1] >= center) << 6)
        lbp |= (np.uint8(working_face[:-2, 2:] >= center) << 5)
        lbp |= (np.uint8(working_face[1:-1, 2:] >= center) << 4)
        lbp |= (np.uint8(working_face[2:, 2:] >= center) << 3)
        lbp |= (np.uint8(working_face[2:, 1:-1] >= center) << 2)
        lbp |= (np.uint8(working_face[2:, :-2] >= center) << 1)
        lbp |= np.uint8(working_face[1:-1, :-2] >= center)
        histogram, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        histogram = histogram.astype(np.float32)
        histogram /= max(float(histogram.sum()), 1.0)
        return histogram

    def _cosine_similarity(self, signature_a: np.ndarray, signature_b: np.ndarray) -> float:
        similarity = float(np.dot(signature_a, signature_b))
        return max(min(similarity, 1.0), 0.0)

    def _pixel_similarity(self, face_a: np.ndarray, face_b: np.ndarray) -> float:
        correlation = float(
            cv2.matchTemplate(
                face_a.astype(np.float32),
                face_b.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )[0][0]
        )
        return max(min(correlation, 1.0), 0.0)

    def _distance_score(
        self,
        raw_distance: float,
        model_name: str = "lbph",
        model_stats: dict[str, dict[str, float | int]] | None = None,
    ) -> float:
        distance_ceiling = self.default_distance_ceilings.get(model_name, self.distance_ceiling)
        if model_stats:
            model_metadata = model_stats.get(model_name, {})
            distance_ceiling = float(model_metadata.get("distance_ceiling", distance_ceiling))
        return round(min(max(raw_distance / max(distance_ceiling, 0.0001), 0.0), 1.0), 4)

    def _student_sample_paths(self, student: Student) -> list[Path]:
        return self._student_sample_paths_for_id(student.student_id)

    def _student_sample_paths_for_id(self, student_id: str) -> list[Path]:
        dataset_dir = self.dataset_root / student_id
        if not dataset_dir.exists():
            return []
        return sorted(dataset_dir.glob("sample_*.png"))

    def _save_capture(self, data_url: str, group: str, identifier: str) -> Path:
        match = re.match(r"^data:image/(png|jpeg|jpg);base64,(.+)$", data_url)
        if not match:
            raise ValueError("Camera image data is invalid.")

        extension = "jpg" if match.group(1) in {"jpeg", "jpg"} else "png"
        capture_dir = self.capture_root / group / identifier
        capture_dir.mkdir(parents=True, exist_ok=True)
        filename = f"capture_{len(list(capture_dir.glob('capture_*'))) + 1:02d}.{extension}"
        capture_path = capture_dir / filename
        capture_path.write_bytes(base64.b64decode(match.group(2)))
        return capture_path
