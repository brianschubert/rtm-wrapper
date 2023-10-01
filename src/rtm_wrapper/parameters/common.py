"""
Parameter definitions.
"""

from __future__ import annotations

import abc
import math

from rtm_wrapper.parameters.base import (
    AbstractParameter,
    FloatArrayField,
    FloatField,
    IntField,
    Parameter,
    ParameterField,
    StrField,
)


class AltitudePredefined(Parameter):
    """Predefined altitude level."""

    name: StrField = StrField(title="Altitude")
    """Name of altitude level."""


class AltitudeKilometers(Parameter):
    """Altitude given in kilometers"""

    value: FloatField = FloatField(title="Altitude", unit="km")
    """Altitude in kilometers."""


class AtmospherePredefined(Parameter):
    """Predefined atmosphere profile."""

    name: StrField = StrField(title="Atmosphere Profile")
    """Name of atmosphere profile."""


class AtmosphereWaterOzone(Parameter):
    """Atmosphere expressed as water and ozone columns."""

    water: FloatField = FloatField(title="Water Column", unit="g/cm^2")
    """Water column in g/cm^2."""
    ozone: FloatField = FloatField(title="Ozone Column", unit="cm-atm")
    """Ozone column in cm-atm."""


class AerosolProfilePredefined(Parameter):
    """Predefined aerosol profile."""

    profile = StrField(title="Aerosol Profile")
    """Name of aerosol profile."""


class AerosolAOTSingleLayer(AerosolProfilePredefined):
    """Aerosol profile consisting of a single layer with a given AOT."""

    height = FloatField(title="Height", unit="km")
    """Heights of each layer in kilometers."""

    aot = FloatField(title="AOT", unit="1")
    """Aerosol optical thickness of each layer."""


class AerosolAOTLayers(AerosolProfilePredefined):
    """Aerosol profile consist of various AOT layers."""

    layers: FloatArrayField = FloatArrayField()
    """
    AOT layers given as Nx2 array, with layer heights in the first column and layer
    AOTs in the second column.
    """


class GroundReflectanceHomogenousUniformLambertian(Parameter):
    """Uniform homogeneous lambertian ground reflectance."""

    reflectance: FloatField = FloatField(title="Reflectance", unit="1")


class GroundReflectanceHomogenousLambertian(Parameter):
    """Homogeneous lambertian ground reflectance."""

    wavelengths: FloatArrayField = FloatArrayField("Wavelength", unit="micrometers")
    """Wavelengths vector."""

    spectrum: FloatArrayField = FloatArrayField("Reflectance", unit="1")
    """Spectrum vector."""


class GroundReflectanceHeterogeneousLambertian(Parameter):
    """Heterogeneous lambertian ground reflectance."""

    target: ParameterField[GroundReflectanceHomogenousLambertian] = ParameterField(
        GroundReflectanceHomogenousLambertian
    )
    """Target spectrum."""

    background: ParameterField[GroundReflectanceHomogenousLambertian] = ParameterField(
        GroundReflectanceHomogenousLambertian
    )
    """Background spectrum."""


class WavelengthFixed(Parameter):
    """Single wavelength."""

    value: FloatField = FloatField(title="Wavelength", unit="micrometers")
    """Wavelength in micrometers."""


class AngleParameter(AbstractParameter):
    """Base class for angle parameters."""

    @abc.abstractmethod
    def as_degrees(self) -> float:
        """Retrieve this angle as degrees."""
        ...


class AngleDegreesParameter(AngleParameter):
    """Angle in degrees."""

    degrees: FloatField = FloatField(title="Angle", unit="degrees")
    """Angle in degrees."""

    def as_degrees(self) -> float:
        return self.degrees


class AngleCosineParameter(AngleParameter):
    """Cosine of angle."""

    cosine: FloatField = FloatField(title="Angle Cosine", unit="1")
    """Angle cosine."""

    def as_degrees(self) -> float:
        m = math  # Save a LOAD_GLOBAL. Probably a flagrantly premmature optiziataion
        return m.degrees(m.acos(self.cosine))


class GeometryAngleDate(Parameter):
    """Geometry description."""

    solar_zenith: ParameterField[AngleParameter] = ParameterField(
        AngleParameter, title="Solar Zenith"  # type: ignore[type-abstract]
    )
    """Solar zenith angle."""

    solar_azimuth: ParameterField[AngleParameter] = ParameterField(
        AngleParameter, title="Solar Azimuth"  # type: ignore[type-abstract]
    )
    """Solar azimuth angle."""

    view_zenith: ParameterField[AngleParameter] = ParameterField(
        AngleParameter, title="View Zenith"  # type: ignore[type-abstract]
    )
    """Target zenith angle."""

    view_azimuth: ParameterField[AngleParameter] = ParameterField(
        AngleParameter, title="View Azimuth"  # type: ignore[type-abstract]
    )
    """Target azimuth angle."""

    day: IntField = IntField(title="Day")
    """Day of month."""

    month: IntField = IntField(title="Month")
    """Month."""
