from .base_bev_backbone import BaseBEVBackbone, BaseBEVBackboneV1, BaseBEVResBackbone
from .ASPP_backbone import ASPP_Backbone
from .bev_backbone_ded import CascadeDEDBackbone
from .NewSSFA import NewSSFA
from .SSFA import SSFA
from .Scnet2DBackbone import Scnet2DBackbone
from .he_bev_backbone import Hed2DBackbone

__all__ = {
    'BaseBEVBackbone': BaseBEVBackbone,
    'BaseBEVBackboneV1': BaseBEVBackboneV1,
    'BaseBEVResBackbone': BaseBEVResBackbone,
    'ASPP_Backbone': ASPP_Backbone,
    'CascadeDEDBackbone': CascadeDEDBackbone,
    "NewSSFA": NewSSFA,
    "SSFA": SSFA,
    "Scnet2DBackbone": Scnet2DBackbone,
    "Hed2DBackbone": Hed2DBackbone,
}
