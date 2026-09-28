import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.media_service import MediaService


class MediaServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.service = MediaService(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def image_bytes(self, size=(1200, 800), image_format="JPEG"):
        stream = io.BytesIO()
        Image.new("RGB", size, "#6f9f8d").save(stream, format=image_format)
        return stream.getvalue()

    def test_avatar_is_cropped_and_manifest_persists(self):
        result = self.service.process_upload(
            "imouto", "avatar", self.image_bytes(), (0.2, 0.0, 0.6, 0.6)
        )
        self.assertEqual((result["width"], result["height"]), (512, 512))
        restored = MediaService(self.temp_dir.name).get_entry("imouto", "avatar")
        self.assertEqual(restored["revision"], result["revision"])
        self.assertTrue(Path(restored["path"]).is_file())

    def test_background_requires_sixteen_by_nine_crop(self):
        with self.assertRaisesRegex(ValueError, "比例"):
            self.service.process_upload(
                "imouto", "background", self.image_bytes(), (0.0, 0.0, 1.0, 1.0)
            )

    def test_rejects_invalid_image_and_out_of_bounds_crop(self):
        with self.assertRaisesRegex(ValueError, "识别"):
            self.service.process_upload("imouto", "avatar", b"not-image", (0, 0, 1, 1))
        with self.assertRaisesRegex(ValueError, "超出"):
            self.service.process_upload(
                "imouto", "avatar", self.image_bytes(), (0.5, 0.5, 0.75, 0.75)
            )

    def test_replacing_media_removes_old_file(self):
        first = self.service.process_upload(
            "imouto", "avatar", self.image_bytes(), (0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0)
        )
        first_path = Path(self.temp_dir.name) / first["filename"]
        second = self.service.process_upload(
            "imouto", "avatar", self.image_bytes(), (0.1, 0.0, 2.0 / 3.0, 2.0 / 3.0)
        )
        self.assertNotEqual(first["revision"], second["revision"])
        self.assertFalse(first_path.exists())


if __name__ == "__main__":
    unittest.main()
