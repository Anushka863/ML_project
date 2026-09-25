"""
ODIR-5K Ophthalmic Multimodal Inference Service.
Phase: ODIR Ophthalmic Branch

Singleton inference service:
1. Loads ODIR preprocessor artifact (models/odir_multimodal/odir_preprocessor.joblib).
2. Loads trained ODIRMultimodalGNN checkpoint (models/odir_multimodal/best_model.pt).
3. Preprocesses input demographics (Age, Sex) and bilateral fundus images (Left & Right).
4. Computes calibrated sigmoid probabilities for 8 ODIR disease heads:
   N (Normal), D (Diabetes), G (Glaucoma), C (Cataract), A (AMD), H (Hypertension), M (Myopia), O (Other).
5. Generates bilateral Grad-CAM saliency heatmaps for both eyes and demographic feature attributions.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing.odir_preprocessor import ODIRPreprocessor, ODIR_DISEASE_LABELS, ODIR_DISEASE_NAMES
from app.models.odir_multimodal_model import ODIRMultimodalGNN
from app.explainability.odir_xai import ODIRBilateralExplainer

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "odir_multimodal" / "best_model.pt"
DEFAULT_PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "odir_multimodal" / "odir_preprocessor.joblib"


class ODIRInferenceService:
    _instance: Optional["ODIRInferenceService"] = None

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        preprocessor_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None
    ):
        self.checkpoint_path = Path(checkpoint_path or DEFAULT_CHECKPOINT_PATH)
        self.preprocessor_path = Path(preprocessor_path or DEFAULT_PREPROCESSOR_PATH)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.preprocessor: Optional[ODIRPreprocessor] = None
        self.model: Optional[ODIRMultimodalGNN] = None
        self.explainer: Optional[ODIRBilateralExplainer] = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads fitted preprocessor and trained neural model checkpoint."""
        if not self.preprocessor_path.exists():
            # Fallback to default preprocessor
            self.preprocessor = ODIRPreprocessor()
        else:
            self.preprocessor = ODIRPreprocessor.load(self.preprocessor_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"ODIR trained checkpoint not found at: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        arch_config = checkpoint.get("arch_config", {})

        self.model = ODIRMultimodalGNN(
            backbone_name=arch_config.get("backbone_name", "resnet18"),
            pretrained=False,
            demo_input_dim=arch_config.get("demo_input_dim", 2),
            demo_emb_dim=arch_config.get("demo_emb_dim", 32),
            single_eye_dim=arch_config.get("single_eye_dim", 64),
            bilateral_img_dim=arch_config.get("bilateral_img_dim", 128),
            fused_dim=arch_config.get("fused_dim", 128),
            gnn_hidden_dim=arch_config.get("gnn_hidden_dim", 64),
            gnn_out_dim=arch_config.get("gnn_out_dim", 32),
            k_neighbors=arch_config.get("k_neighbors", 5),
            disease_targets=checkpoint.get("disease_targets", ODIR_DISEASE_LABELS)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.explainer = ODIRBilateralExplainer(self.model, device=self.device)

    @classmethod
    def get_instance(cls) -> "ODIRInferenceService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def predict(
        self,
        age: float,
        sex: Union[str, int],
        left_image: Union[str, Path, Image.Image],
        right_image: Union[str, Path, Image.Image],
        include_xai: bool = True
    ) -> Dict[str, Any]:
        """
        Runs full multimodal ophthalmic risk assessment.
        """
        if self.model is None or self.preprocessor is None:
            self._load_artifacts()

        # 1. Preprocess demographics
        demo_tensor = self.preprocessor.preprocess_demographics(age=age, sex=sex).to(self.device)

        # 2. Preprocess bilateral images
        left_tensor, right_tensor = self.preprocessor.preprocess_bilateral_images(
            left_img=left_image,
            right_img=right_image,
            is_training=False
        )
        left_tensor = left_tensor.to(self.device)
        right_tensor = right_tensor.to(self.device)

        # 3. Model Forward Pass
        with torch.no_grad():
            probs_dict = self.model.predict_probabilities(demo_tensor, left_tensor, right_tensor)

        # 4. Format 8 disease results
        disease_results = {}
        for code in ODIR_DISEASE_LABELS:
            prob = float(probs_dict[code].item())
            prob_pct = round(prob * 100.0, 1)
            
            if prob >= 0.65:
                risk_level = "High Probability"
                status = "Elevated Model Probability"
            elif prob >= 0.35:
                risk_level = "Moderate Probability"
                status = "Moderate Model Probability"
            else:
                risk_level = "Low Probability"
                status = "Low Model Probability"

            disease_results[code] = {
                "code": code,
                "name": ODIR_DISEASE_NAMES[code],
                "probability": round(prob, 4),
                "probability_pct": prob_pct,
                "risk_level": risk_level,
                "status": status,
                "confidence": round(abs(prob - 0.5) * 200, 1)  # 0 - 100% confidence from decision boundary
            }

        # Highest risk disease (excluding 'N' for target saliency)
        non_normal_diseases = {k: v for k, v in disease_results.items() if k != "N"}
        top_disease_code = max(non_normal_diseases.keys(), key=lambda k: non_normal_diseases[k]["probability"])
        
        # 5. Explainability (Grad-CAM + Demographics)
        xai_data = None
        if include_xai and self.explainer is not None:
            try:
                xai_data = self.explainer.explain_patient(
                    demo_tensor=demo_tensor,
                    left_tensor=left_tensor,
                    right_tensor=right_tensor,
                    top_disease=top_disease_code
                )
            except Exception as e:
                xai_data = {
                    "error": f"Explainability computation failed: {str(e)}",
                    "disclaimer": "Heatmap unavailable."
                }

        # Overall summary
        max_prob_disease = max(disease_results.keys(), key=lambda k: disease_results[k]["probability"])
        overall_pred = f"Primary Finding: {disease_results[max_prob_disease]['name']} ({disease_results[max_prob_disease]['probability_pct']}%)"

        return {
            "status": "success",
            "branch": "ODIR-5K Ophthalmic Multimodal Branch",
            "model_name": "ODIRMultimodalGNN (ResNet-18 Bilateral Vision + Demographic MLP + Patient Similarity Graph)",
            "prediction": overall_pred,
            "patient_demographics": {
                "age": float(age),
                "sex": "Male" if str(sex).lower() in ["male", "m", "1"] else "Female"
            },
            "diseases": disease_results,
            "xai": xai_data,
            "disclaimer": "This AI-generated ophthalmic risk assessment is strictly for clinical decision support research and is not a replacement for professional ophthalmological examination."
        }
