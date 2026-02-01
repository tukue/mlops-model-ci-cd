import os
import json
import joblib
from datetime import datetime
from pathlib import Path

class ModelRegistry:
    def __init__(self, registry_path="artifacts/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(exist_ok=True)
        self.metadata_file = self.registry_path / "metadata.json"
    
    def save_model(self, model, metrics, version=None):
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.registry_path / f"model_v{version}.joblib"
        joblib.dump(model, model_path)
        
        metadata = self._load_metadata()
        metadata[version] = {
            "path": str(model_path),
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
            "is_active": False
        }
        self._save_metadata(metadata)
        return version
    
    def set_active_model(self, version):
        metadata = self._load_metadata()
        for v in metadata:
            metadata[v]["is_active"] = (v == version)
        self._save_metadata(metadata)
        
        # Create symlink to active model
        active_path = Path("artifacts/model.joblib")
        if active_path.exists():
            active_path.unlink()
        active_path.symlink_to(metadata[version]["path"])
    
    def get_active_model(self):
        metadata = self._load_metadata()
        for version, info in metadata.items():
            if info.get("is_active"):
                return version, joblib.load(info["path"])
        return None, None
    
    def list_models(self):
        return self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)