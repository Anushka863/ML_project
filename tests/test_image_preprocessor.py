"""
Test suite for ImagePreprocessor (Phase 4 verification)
"""
import sys
import tempfile
import torch
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.preprocessing.image_preprocessor import ImagePreprocessor


def test_image_preprocessor():
    print("=" * 70)
    print("🧪 TESTING IMAGE PREPROCESSOR (PHASE 4 VERIFICATION)")
    print("=" * 70)

    preprocessor = ImagePreprocessor(image_size=(224, 224))

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Test Valid RGB Image
        valid_img_path = tmp_path / "valid_eye.png"
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        img.save(valid_img_path)

        tensor_val = preprocessor.load_and_preprocess(valid_img_path, is_training=False)
        tensor_train = preprocessor.load_and_preprocess(valid_img_path, is_training=True)

        print(f"✅ Validation Tensor Shape: {tensor_val.shape}")
        print(f"✅ Training Tensor Shape:   {tensor_train.shape}")

        assert tensor_val.shape == (1, 3, 224, 224), f"Unexpected shape {tensor_val.shape}"
        assert tensor_train.shape == (1, 3, 224, 224), f"Unexpected shape {tensor_train.shape}"

        # 2. Test Corrupted / Bad File
        corrupted_path = tmp_path / "corrupted.png"
        with open(corrupted_path, "wb") as f:
            f.write(b"NOT_A_REAL_IMAGE_DATA")

        corrupted_passed = False
        try:
            preprocessor.load_and_preprocess(corrupted_path)
        except ValueError as e:
            corrupted_passed = True
            print(f"✅ Corrupted image correctly caught & rejected: {e}")

        assert corrupted_passed, "Corrupted image was not rejected!"

        # 3. Test Unsupported Extension
        txt_path = tmp_path / "data.txt"
        with open(txt_path, "w") as f:
            f.write("text data")

        unsupported_passed = False
        try:
            preprocessor.load_and_preprocess(txt_path)
        except ValueError as e:
            unsupported_passed = True
            print(f"✅ Unsupported extension correctly rejected: {e}")

        assert unsupported_passed, "Unsupported extension was not rejected!"

    print("\n🎉 ALL IMAGE PREPROCESSOR TESTS PASSED!")


if __name__ == "__main__":
    test_image_preprocessor()
