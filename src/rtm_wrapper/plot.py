from __future__ import annotations

from typing import Hashable

import matplotlib.pyplot as plt
import xarray as xr
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_sweep_single(sweep_variable: xr.DataArray, *, ax: Axes | None = None) -> None:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if sweep_variable.ndim != 1:
        raise ValueError(
            f"only single dimensional sweeps supported, got ndim={sweep_variable.ndim}"
        )

    (sweep_dim,) = sweep_variable.dims

    sweep_coords = sweep_variable.coords[sweep_dim]

    ax.plot(sweep_coords.values, sweep_variable.values)
    ax.set_xlabel(_coords_axes_label(sweep_coords))
    ax.set_ylabel(_coords_axes_label(sweep_variable))


def plot_sweep_legend(sweep_variable: xr.DataArray) -> None:
    pass


def _coords_with_dims(arr: xr.DataArray, dims: tuple[Hashable, ...]) -> list[Hashable]:
    return [name for name, coord in arr.coords.items() if coord.dims == dims]


def _coords_axes_label(coords: xr.DataArray) -> str:
    base_label = coords.attrs.get("title", coords.name)
    if "unit" in coords.attrs:
        unit_str = coords.attrs["unit"].replace("-", r"\cdot{}")
        return rf"{base_label} (${unit_str}$)"
    else:
        return base_label
