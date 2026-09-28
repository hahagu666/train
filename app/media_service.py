import io
import json
import os
import threading
import uuid
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import USER_MEDIA_DIR


class MediaService:
    KINDS = {
        "avatar": {"ratio": 1.0, "size": (512, 512)},
        "background": {"ratio": 16.0 / 9.0, "size": (1920, 1080)},
    }
    MAX_BYTES = 15 * 1024 * 1024
    MAX_PIXELS = 48_000_000
    MIN_SIDE = 128

    def __init__(self, root=USER_MEDIA_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()

    def _load_manifest(self):
        if not self.manifest_path.exists():
            return {}
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_manifest(self, manifest):
        temp_path = self.manifest_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, self.manifest_path)

    def get_entry(self, character_id, kind):
        if kind not in self.KINDS:
            return None
        entry = self._manifest.get(character_id, {}).get(kind)
        if not isinstance(entry, dict):
            return None
        filename = entry.get("filename", "")
        path = self.root / filename
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            return None
        if not path.is_file():
            return None
        return dict(entry, path=str(path))

    def apply_to_character(self, character):
        for kind in self.KINDS:
            entry = self.get_entry(character.id, kind)
            if entry:
                setattr(character, kind, "/api/media/{}/{}".format(character.id, kind))
                setattr(character, kind + "_revision", int(entry.get("revision", 0)))
        return character

    def process_upload(self, character_id, kind, payload, crop):
        if kind not in self.KINDS:
            raise ValueError("媒体类型无效")
        if not payload or len(payload) > self.MAX_BYTES:
            raise ValueError("图片大小无效或超过 15 MB")
        x, y, width, height = [float(value) for value in crop]
        if min(x, y, width, height) < 0 or width <= 0 or height <= 0:
            raise ValueError("裁剪区域无效")
        if x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("裁剪区域超出图片")
        expected_ratio = self.KINDS[kind]["ratio"]
        if abs((width / height) - expected_ratio) > 0.015:
            raise ValueError("裁剪比例不符合要求")

        try:
            with Image.open(io.BytesIO(payload)) as source:
                if getattr(source, "is_animated", False) or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("不支持动态图")
                if source.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("仅支持 JPEG、PNG 和 WebP")
                pixel_count = source.width * source.height
                if pixel_count > self.MAX_PIXELS:
                    raise ValueError("图片像素过大")
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise ValueError("无法识别或读取图片") from exc

        left = round(x * image.width)
        top = round(y * image.height)
        right = round((x + width) * image.width)
        bottom = round((y + height) * image.height)
        if min(right - left, bottom - top) < self.MIN_SIDE:
            raise ValueError("裁剪区域过小")

        output = ImageOps.fit(
            image.crop((left, top, right, bottom)),
            self.KINDS[kind]["size"],
            method=Image.Resampling.LANCZOS,
        )
        revision = uuid.uuid4().int >> 64
        filename = "{}_{}_{}.jpg".format(character_id, kind, revision)
        temp_path = self.root / (filename + ".tmp")
        final_path = self.root / filename
        output.save(temp_path, format="JPEG", quality=90, optimize=True)

        with self._lock:
            old_entry = self._manifest.get(character_id, {}).get(kind)
            os.replace(temp_path, final_path)
            updated = json.loads(json.dumps(self._manifest))
            updated.setdefault(character_id, {})[kind] = {
                "filename": filename,
                "revision": revision,
                "width": output.width,
                "height": output.height,
            }
            try:
                self._save_manifest(updated)
            except Exception:
                final_path.unlink(missing_ok=True)
                raise
            self._manifest = updated
            if isinstance(old_entry, dict):
                old_path = self.root / old_entry.get("filename", "")
                if old_path != final_path:
                    old_path.unlink(missing_ok=True)

        return dict(updated[character_id][kind], url="/api/media/{}/{}".format(character_id, kind))

    def delete_character_media(self, character_id):
        with self._lock:
            entries = self._manifest.get(character_id)
            if not entries:
                return
            updated = json.loads(json.dumps(self._manifest))
            updated.pop(character_id, None)
            self._save_manifest(updated)
            self._manifest = updated
            for entry in entries.values():
                if isinstance(entry, dict):
                    (self.root / entry.get("filename", "")).unlink(missing_ok=True)
