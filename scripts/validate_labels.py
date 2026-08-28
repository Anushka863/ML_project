"""
Disease Label Compatibility Validation Script
Phase 1 — Step 3: Validate whether clinical disease targets and image labels represent identical prediction tasks.
"""
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent


LABEL_MAPPINGS_AUDIT = [
    {
        "disease": "Diabetes",
        "clinical_target": "Systemic Diabetes Mellitus (Glucose ≥ 126 mg/dL, HbA1c ≥ 6.5%, or diagnosed)",
        "image_modality": "APTOS 2019 Retinal Fundus Images",
        "image_labels": "Diabetic Retinopathy (DR) Severity Grade (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)",
        "compatible": False,
        "reason": "Retinal fundus images grade microvascular eye organ damage (Diabetic Retinopathy), which is a complication of diabetes, NOT systemic blood glucose / diabetes diagnosis itself. They are clinically correlated proxy tasks, but not identical labels."
    },
    {
        "disease": "Chronic Kidney Disease (CKD)",
        "clinical_target": "Systemic CKD (eGFR < 60 mL/min/1.73m² or Serum Creatinine / BUN elevation)",
        "image_modality": "CT Kidney Scans",
        "image_labels": "Structural CT Lesions (Normal, Cyst, Tumor, Stone)",
        "compatible": False,
        "reason": "Kidney CT scans detect anatomical lesions (stones, cysts, tumors), whereas clinical CKD diagnosis measures functional nephron filtration (eGFR, Creatinine). A patient with a kidney cyst may have normal eGFR, and a patient with severe CKD may have structurally normal CT scans."
    },
    {
        "disease": "Heart Disease",
        "clinical_target": "Ischemic / Coronary Heart Disease (History of MI, Angina, Hypertension, High Risk)",
        "image_modality": "PTB-XL ECG Signals",
        "image_labels": "Electrophysiological Arrhythmias / Conduction Abnormalities",
        "compatible": False,
        "reason": "ECG measures electrical signal abnormalities (arrhythmias, bundle branch blocks), which are proxy markers and not directly equivalent to NHANES clinical survey history of ischemic heart disease."
    }
]


def validate_label_compatibility():
    print("=" * 70)
    print("🏷️ DISEASE LABEL COMPATIBILITY VALIDATION")
    print("=" * 70)
    
    all_compatible = True
    
    for item in LABEL_MAPPINGS_AUDIT:
        print(f"\n🔍 Target Condition: {item['disease']}")
        print(f"   • Clinical Target Definition: {item['clinical_target']}")
        print(f"   • Image Modality & Labels:   {item['image_modality']}")
        print(f"                                {item['image_labels']}")
        print(f"   • Task Compatibility:        {'VALID ✅' if item['compatible'] else 'INCOMPATIBLE / PROXY ONLY ⚠️'}")
        print(f"   • Clinical Rationale:        {item['reason']}")
        
        if not item['compatible']:
            all_compatible = False
            
    print("\n" + "-" * 70)
    print("📊 LABEL COMPATIBILITY SUMMARY")
    print("-" * 70)
    print(f"• Medically Related Labels Evaluated: {len(LABEL_MAPPINGS_AUDIT)}")
    print(f"• Identical Task Mappings Found:     0")
    print(f"• Label Compatibility Status:        INVALID ❌ (Proxy / Correlated tasks, not equivalent targets)")
    print("-" * 70)
    
    print("\n🚨 CRITICAL LESSON & AUDIT INSTRUCTION:")
    print("Medically related labels MUST NOT be assumed to be equivalent target predictions.")
    print("Direct multi-modal label training on merged target heads requires exact target alignment.")
    
    return {
        "label_compatibility_valid": all_compatible,
        "audit_mappings": LABEL_MAPPINGS_AUDIT
    }


if __name__ == "__main__":
    validate_label_compatibility()
