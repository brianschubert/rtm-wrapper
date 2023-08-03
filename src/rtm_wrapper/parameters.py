"""
Model agnostic descriptions of simulation input parameters.
"""

from __future__ import annotations

import copy
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import numpy as np
from typing_extensions import NotRequired


class MetadataDict(TypedDict):
    title: NotRequired[str]
    unit: NotRequired[str]


class Parameter:
    def set(
        self, param: str | None = None, value: Any | None = None, /, **kwargs: Any
    ) -> None:
        if kwargs:
            if param is not None:
                raise ValueError(
                    "kwargs must not be passed when positional arguments are used"
                )
            for param_path, param_arg in kwargs.items():
                self.set(param_path, param_arg)
            return

        if param is None:
            raise ValueError(
                "parameter name must be specified when no kwargs are given"
            )

        current_attr, _sep, sub_param_path = param.partition("__")

        if sub_param_path:
            sub_param = getattr(self, current_attr)
            sub_param.set(sub_param_path, value)
        else:
            current_value = getattr(self, current_attr)
            old_is_param = isinstance(current_value, Parameter)
            new_is_param = isinstance(value, Parameter)
            if old_is_param or new_is_param:
                if not old_is_param or not new_is_param:
                    raise ValueError(
                        f"attempt to change parameter status - old is {type(current_value).__name__}, new is {type(value).__name__}"
                    )
            setattr(self, current_attr, value)

    def get_metadata(self, param_path: str) -> MetadataDict:
        current_attr, _sep, sub_param_path = param_path.partition("__")
        if sub_param_path:
            try:
                sub_param: Parameter = getattr(self, current_attr)
            except AttributeError:
                raise AttributeError(
                    f"unable to resolve sub-parameter '{current_attr}' on {self.__class__.__name__}"
                )
            return sub_param.get_metadata(sub_param_path)

        try:
            all_hints = typing.get_type_hints(self.__class__, include_extras=True)
            attr_hints = all_hints[current_attr]
        except KeyError:
            raise AttributeError(
                f"unable to resolve terminal parameter '{current_attr}' on {self.__class__.__name__}"
            )
        if typing.get_origin(attr_hints) is Annotated:
            metadata: MetadataDict = typing.get_args(attr_hints)[1]
            return metadata
        else:
            return {}

    def replace(self, **kwargs: Any) -> Parameter:
        duplicate = copy.deepcopy(self)
        duplicate.set(**kwargs)
        return duplicate


@dataclass
class AltitudePredefined(Parameter):
    name: Annotated[Literal["sealevel", "satellite"], MetadataDict(title="Altitude")]


@dataclass
class AltitudeKilometers(Parameter):
    value: Annotated[float, MetadataDict(title="Altitude")]


@dataclass
class AtmospherePredefined(Parameter):
    name: Annotated[
        Literal[
            "NoGaseousAbsorption",
            "Tropical",
            "MidlatitudeSummer",
            "MidlatitudeWinter",
            "SubarcticSummer",
            "SubarcticWinter",
        ],
        MetadataDict(title="Atmosphere Profile"),
    ]


@dataclass
class AtmosphereWaterOzone(Parameter):
    water: Annotated[float, MetadataDict(title="Water Column", unit="g/cm^2")]
    ozone: Annotated[float, MetadataDict(title="Ozone Column", unit="cm-atm")]


@dataclass
class AtmosphereAotLayers(AtmospherePredefined):
    layers: np.ndarray


@dataclass
class AerosolProfilePredefined(Parameter):
    name: Annotated[
        Literal["Maritime", "Urban", "Continental"],
        MetadataDict(title="Aerosol Profile"),
    ]


@dataclass
class GroundReflectanceHomogenousUniformLambertian(Parameter):
    reflectance: Annotated[float, MetadataDict(title="Reflectance")]


@dataclass
class GroundReflectanceHomogenousLambertian(Parameter):
    wavelengths: np.ndarray
    spectrum: np.ndarray


@dataclass
class GroundReflectanceHeterogeneousUniformLambertian(Parameter):
    target: float
    background: float


@dataclass
class GroundReflectanceHeterogeneousLambertian(Parameter):
    target: GroundReflectanceHomogenousLambertian
    background: GroundReflectanceHomogenousLambertian


@dataclass
class WavelengthFixed(Parameter):
    value: Annotated[float, MetadataDict(title="Wavelength", unit="micrometers")]
