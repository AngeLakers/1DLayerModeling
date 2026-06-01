from .attenuation import AttenuationLaw, ConstantAttenuation, PowerLawAttenuation
from .adhesives import (
    A1_DEFAULT_ADHESIVE_PRIOR,
    ADHESIVE_PRIORS,
    NOA60_HALDREN_2019_PRIOR,
    AdhesiveLayerPrior,
    make_a1_default_adhesive_layer,
    make_a1_default_adhesive_material,
)
from .media import HalfSpaceMedium
from .model import Layer, InterfaceSpring, LaminatedStack
from .model_checks import (
    ZeroThicknessInterfaceApplicability,
    check_layer_as_zero_thickness_interface,
    check_zero_thickness_interface_applicability,
    classify_zero_thickness_interface_ratio,
    zero_thickness_interface_ratio,
)
from .solver import FrequencyResponseResult

__version__ = "1.2.2"

__all__ = [
    'AttenuationLaw',
    'A1_DEFAULT_ADHESIVE_PRIOR',
    'ADHESIVE_PRIORS',
    'NOA60_HALDREN_2019_PRIOR',
    'AdhesiveLayerPrior',
    'ConstantAttenuation',
    'PowerLawAttenuation',
    'HalfSpaceMedium',
    'Layer',
    'InterfaceSpring',
    'LaminatedStack',
    'ZeroThicknessInterfaceApplicability',
    'check_layer_as_zero_thickness_interface',
    'check_zero_thickness_interface_applicability',
    'classify_zero_thickness_interface_ratio',
    'zero_thickness_interface_ratio',
    'make_a1_default_adhesive_layer',
    'make_a1_default_adhesive_material',
    'FrequencyResponseResult',
    '__version__',
]
