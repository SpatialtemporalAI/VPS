from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from hloc import extract_features, extractors
from hloc.utils.base_model import dynamic_load


class VPRModel:
    """Global descriptor encoder.

    This class owns only the retrieval backbone and descriptor extraction.
    """

    def __init__(self, config: Dict):
        self.device = config["system"]["device"]
        self.method = config["vpr"].get("method", "megaloc")
        if self.method == "netvlad":
            self.retrieval_conf = extract_features.confs["netvlad"]
        elif self.method == "megaloc":
            self.retrieval_conf = extract_features.confs["megaloc"]
        else:
            raise ValueError(f"Unsupported VPR method: {self.method}")

        self.model = dynamic_load(extractors, self.retrieval_conf["model"]["name"])(
            self.retrieval_conf["model"]
        ).eval().to(self.device)

    def extract_global_descriptors(
        self, images: Union[str, Path, List[Union[str, Path]]], output_path: Path
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return extract_features.main(
            self.retrieval_conf,
            images,
            feature_path=output_path,
            overwrite=True,
            model=self.model,
        )
