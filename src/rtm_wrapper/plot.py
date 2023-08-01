from __future__ import annotations

from typing import Any, Hashable

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_sweep_single(
    sweep_variable: xr.DataArray, *, ax: Axes | None = None
) -> tuple[Figure, Axes]:
    sweep_variable = sweep_variable.squeeze(drop=True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if sweep_variable.ndim != 1:
        raise ValueError(
            f"only single dimensional sweeps supported, "
            f"got ndim={sweep_variable.ndim} (after squeezing)"
        )

    (sweep_dim,) = sweep_variable.dims

    sweep_coords = sweep_variable.coords[sweep_dim]

    ax.plot(sweep_coords.values, sweep_variable.values)
    ax.set_xlabel(_coords_axes_label(sweep_coords))
    ax.set_ylabel(_coords_axes_label(sweep_variable))

    return fig, ax


def plot_sweep_legend(
    sweep_variable: xr.DataArray,
    *,
    ax: Axes | None = None,
    legend_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, Axes]:
    sweep_variable = sweep_variable.squeeze(drop=True)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if sweep_variable.ndim != 2:
        raise ValueError(
            f"only two dimensional sweeps supported, "
            f"got ndim={sweep_variable.ndim} (after squeezing)"
        )

    legend_dim, axes_dim = sweep_variable.dims

    legend_coords = sweep_variable.coords[legend_dim]
    axes_coords = sweep_variable.coords[axes_dim]

    for legend_idx, legend_label in enumerate(legend_coords.values):
        if isinstance(legend_label, float):
            legend_label = f"{legend_label:.3f}"
        ax.plot(
            axes_coords.values,
            sweep_variable.values[legend_idx, :],
            label=str(legend_label),
        )
    ax.set_xlabel(_coords_axes_label(axes_coords))
    ax.set_ylabel(_coords_axes_label(sweep_variable))

    if legend_kwargs is None:
        legend_kwargs = {}
    legend_kwargs.setdefault("title", _coords_axes_label(legend_coords))
    ax.legend(**legend_kwargs)

    return fig, ax


def plot_sweep_grid(
    sweep_variable: xr.DataArray,
    *,
    fig: Figure | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    sweep_variable = sweep_variable.squeeze(drop=True)

    if sweep_variable.ndim != 3:
        raise ValueError(
            f"only three dimensional sweeps supported, "
            f"got ndim={sweep_variable.ndim} (after squeezing)"
        )

    grid_y_dim, grid_x_dim, axes_dim = sweep_variable.dims

    grid_y_coords = sweep_variable.coords[grid_y_dim]
    grid_x_coords = sweep_variable.coords[grid_x_dim]
    axes_coords = sweep_variable.coords[axes_dim]

    if fig is None:
        fig = plt.figure()

    subplot_args = subplot_kwargs if subplot_kwargs is not None else {}
    axs = fig.subplots(
        nrows=grid_y_coords.size, ncols=grid_x_coords.size, **subplot_args
    )

    for ax, idx in zip(axs.flat, np.ndindex(grid_y_coords.size, grid_x_coords.size)):
        ax.plot(
            axes_coords.values,
            sweep_variable.values[idx[0], idx[1], :],
        )

    y_prefix = _coords_axes_label(grid_y_coords, include_units=False)
    for ax, label in zip(axs[:, 0], grid_y_coords.values):
        ax.set_ylabel(f"{y_prefix}={label}")
    x_prefix = _coords_axes_label(grid_x_coords, include_units=False)
    for ax, label in zip(axs[0, :], grid_x_coords.values):
        ax.set_title(f"{x_prefix}={label}")

    fig.supxlabel(_coords_axes_label(axes_coords))
    fig.supylabel(_coords_axes_label(sweep_variable))

    return fig, axs


def _coords_with_dims(arr: xr.DataArray, dims: tuple[Hashable, ...]) -> list[Hashable]:
    return [name for name, coord in arr.coords.items() if coord.dims == dims]


def _coords_axes_label(coords: xr.DataArray, include_units: bool = True) -> str:
    base_label = coords.attrs.get("title")
    if base_label is None:
        # Title missing OR set to None.
        base_label = coords.name
    if include_units and coords.attrs.get("unit") is not None:
        # Unit exists and was not set to None.
        unit_str = coords.attrs["unit"].replace("-", r"\cdot{}")
        return rf"{base_label} (${unit_str}$)"
    else:
        return base_label
