"""
Test suite for ClinicalEncoder (Phase 5) and ImageEncoder (Phase 6)
"""
import sys
import torch
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.models.clinical_encoder import ClinicalEncoder
from app.models.image_encoder import ImageEncoder


def test_clinical_encoder():
    print("=" * 70)
    print("🧪 TESTING CLINICAL ENCODER (PHASE 5 VERIFICATION)")
    print("=" * 70)

    batch_size = 16
    input_dim = 13
    embedding_dim = 64

    model = ClinicalEncoder(input_dim=input_dim, embedding_dim=embedding_dim)
    dummy_input = torch.randn(batch_size, input_dim)

    out = model(dummy_input)
    print(f"✅ Input Shape:  {dummy_input.shape}")
    print(f"✅ Output Shape: {out.shape}")

    assert out.shape == (batch_size, embedding_dim), f"Expected shape ({batch_size}, {embedding_dim}), got {out.shape}"
    print("🎉 CLINICAL ENCODER TEST PASSED!")


def test_image_encoder():
    print("\n" + "=" * 70)
    print("🧪 TESTING IMAGE ENCODER (PHASE 6 VERIFICATION)")
    print("=" * 70)

    batch_size = 4
    embedding_dim = 64

    for backbone in ["resnet18", "efficientnet_b0"]:
        encoder = ImageEncoder(backbone_name=backbone, pretrained=False, embedding_dim=embedding_dim)
        dummy_img = torch.randn(batch_size, 3, 224, 224)

        out = encoder(dummy_img)
        print(f"✅ Backbone '{backbone}' Output Shape: {out.shape}")
        assert out.shape == (batch_size, embedding_dim), f"Expected ({batch_size}, {embedding_dim}), got {out.shape}"

    print("🎉 IMAGE ENCODER TEST PASSED!")


if __name__ == "__main__":
    test_clinical_encoder()
    test_image_encoder()
