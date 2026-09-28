import json
import os

from .config import DATA_DIR
from .models import AppSettings, UpdateAppSettingsRequest


SETTINGS_PATH = os.path.join(DATA_DIR, "app_settings.json")


class AppSettingsManager:
    def __init__(self):
        self.settings = AppSettings()
        self._load()

    def _load(self):
        if not os.path.exists(SETTINGS_PATH):
            return
        with open(SETTINGS_PATH, "r", encoding="utf-8") as file:
            self.settings = AppSettings(**json.load(file))

    def get(self) -> AppSettings:
        return self.settings.model_copy(deep=True)

    def update(self, request: UpdateAppSettingsRequest) -> AppSettings:
        data = self.settings.model_dump()
        data.update(request.model_dump(exclude_unset=True))
        self.settings = AppSettings(**data)
        self._save()
        return self.get()

    def _save(self):
        temp_path = f"{SETTINGS_PATH}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(self.settings.model_dump(), file, ensure_ascii=False, indent=2)
        os.replace(temp_path, SETTINGS_PATH)
