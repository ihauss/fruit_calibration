from .base import SegmentationEngine
from .hsv_engine import HSVSegmentation
from .unet_engine import UNetSegmentation

class SegmentationFactory:
    @staticmethod
    def create(engine_type: str, **kwargs) -> SegmentationEngine:
        if engine_type == "hsv":
            # Paramètres optionnels pour HSV
            return HSVSegmentation(**kwargs)
        elif engine_type == "unet":
            # checkpoint_path est obligatoire
            return UNetSegmentation(**kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")
