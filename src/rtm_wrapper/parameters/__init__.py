"""
Model agnostic descriptions of simulation input parameters.

Common RTM parameters are defined in :mod:`.common`.
"""

from __future__ import annotations

from .base import (
    AbstractParameter,
    AbstractParameterMeta,
    Field,
    FloatArrayField,
    FloatField,
    IntField,
    Parameter,
    ParameterField,
    ParameterMeta,
    StrField,
)
from .common import (
    AerosolAOTLayers,
    AerosolAOTSingleLayer,
    AerosolProfilePredefined,
    AltitudeKilometers,
    AltitudePredefined,
    AngleCosineParameter,
    AngleDegreesParameter,
    AngleParameter,
    AtmospherePredefined,
    AtmosphereWaterOzone,
    GeometryAngleDate,
    GroundReflectanceHeterogeneousLambertian,
    GroundReflectanceHomogenousLambertian,
    GroundReflectanceHomogenousUniformLambertian,
    WavelengthFixed,
)
from .util import MetadataDict, ParameterError, UnsetParameterError

__all__ = [
    "AbstractParameter",
    "AbstractParameterMeta",
    "Field",
    "FloatArrayField",
    "FloatField",
    "IntField",
    "Parameter",
    "ParameterField",
    "ParameterMeta",
    "StrField",
    "AerosolAOTLayers",
    "AerosolAOTSingleLayer",
    "AerosolProfilePredefined",
    "AltitudeKilometers",
    "AltitudePredefined",
    "AngleCosineParameter",
    "AngleDegreesParameter",
    "AngleParameter",
    "AtmospherePredefined",
    "AtmosphereWaterOzone",
    "GeometryAngleDate",
    "GroundReflectanceHeterogeneousLambertian",
    "GroundReflectanceHomogenousLambertian",
    "GroundReflectanceHomogenousUniformLambertian",
    "WavelengthFixed",
    "MetadataDict",
    "ParameterError",
    "UnsetParameterError",
]
