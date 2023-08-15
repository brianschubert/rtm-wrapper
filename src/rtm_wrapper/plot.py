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

    (sweep_dim,) = sweep_variable.indexes.dims

    sweep_coords = sweep_variable.coords[sweep_dim]

    ax.plot(sweep_coords.values, sweep_variable.values)
    ax.set_xlabel(_coords_axes_label(sweep_coords))
    ax.set_ylabel(_coords_axes_label(sweep_variable))

    return fig, ax


def plot_sweep_legend(
    sweep_variable: xr.DataArray,
    *,
    ax: Axes | None = None,
    xaxis_dim: str | None = None,
    legend_dim: str | None = None,
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

    xaxis_dim, legend_dim = _resolve_dims(
        list(sweep_variable.indexes.dims), [xaxis_dim, legend_dim]  # type: ignore
    )

    legend_coords = sweep_variable.coords[legend_dim]
    axes_coords = sweep_variable.coords[xaxis_dim]

    for legend_idx, legend_label in enumerate(legend_coords.values):
        if isinstance(legend_label, float):
            legend_label = f"{legend_label:.3f}"
        print(
            axes_coords.values,
            sweep_variable.isel({legend_dim: legend_idx}, drop=True),
        )
        ax.plot(
            axes_coords.values,
            sweep_variable.isel({legend_dim: legend_idx}, drop=True),
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
    xaxis_dim: str | None = None,
    grid_y_dim: str | None = None,
    grid_x_dim: str | None = None,
    fig: Figure | None = None,
    allow_big_grid: bool = False,
    subplot_kwargs: dict[str, Any] | None = None,
) -> tuple[Figure, Axes | np.ndarray]:
    sweep_variable = sweep_variable.squeeze(drop=True)

    if sweep_variable.ndim != 3:
        raise ValueError(
            f"only three dimensional sweeps supported, "
            f"got ndim={sweep_variable.ndim} (after squeezing)"
        )

    xaxis_dim, grid_y_dim, grid_x_dim = _resolve_dims(
        list(sweep_variable.indexes.dims), [xaxis_dim, grid_y_dim, grid_x_dim]  # type: ignore
    )

    axes_coords = sweep_variable.coords[xaxis_dim]
    grid_y_coords = sweep_variable.coords[grid_y_dim]
    grid_x_coords = sweep_variable.coords[grid_x_dim]

    # Safety check to prevent accidental creation of massive figures.
    if not allow_big_grid and (grid_x_coords.size >= 10 or grid_y_coords.size >= 10):
        raise ValueError(
            f"request plot grid size ({grid_y_coords.size}, {grid_x_coords.size})"
            f" is too big. Pass 'allow_big_grid=True' to enable plotting of large grids."
        )

    if fig is None:
        fig = plt.figure()

    subplot_args = subplot_kwargs if subplot_kwargs is not None else {}
    axs = fig.subplots(
        nrows=grid_y_coords.size, ncols=grid_x_coords.size, **subplot_args
    )

    for ax, idx in zip(axs.flat, np.ndindex(grid_y_coords.size, grid_x_coords.size)):
        ax.plot(
            axes_coords.values,
            sweep_variable.isel({grid_y_dim: idx[0], grid_x_dim: idx[1]}),
        )

    y_prefix = _coords_axes_label(grid_y_coords, include_units=False)
    for ax, label in zip(axs[:, 0], grid_y_coords.values):
        sep = "=" if len(y_prefix) + len(label) < 18 else "=\n"
        ax.set_ylabel(f"{y_prefix}{sep}{label}")
    x_prefix = _coords_axes_label(grid_x_coords, include_units=False)
    for ax, label in zip(axs[0, :], grid_x_coords.values):
        sep = "=" if len(x_prefix) + len(label) < 18 else "=\n"
        ax.set_title(f"{x_prefix}{sep}{label}")

    fig.supxlabel(_coords_axes_label(axes_coords))
    fig.supylabel(_coords_axes_label(sweep_variable))

    return fig, axs


def _coords_with_dims(arr: xr.DataArray, dims: tuple[Hashable, ...]) -> list[Hashable]:
    return [name for name, coord in arr.coords.items() if coord.dims == dims]


def _coords_axes_label(coords: xr.DataArray, include_units: bool = True) -> str:
    base_label: str | None = coords.attrs.get("title")
    if base_label is None:
        # Title missing OR set to None.
        base_label = str(coords.name)
    if include_units and coords.attrs.get("unit") is not None:
        # Unit exists and was not set to None.
        unit_str = coords.attrs["unit"].replace("-", r"\cdot{}")
        return rf"{base_label} (${unit_str}$)"
    else:
        return base_label


def _resolve_dims(valid_dims: list[str], given_dims: list[str | None]) -> list[str]:
    if len(valid_dims) != len(given_dims):
        raise ValueError(
            f"dimension count mismatch - there are {len(valid_dims)} valid dimension,"
            f"but {len(given_dims)} resolved dimensions were requested"
        )

    # Assume valid dims are unique.
    # Reverse so the left-most entries are popped first.
    unused_dims: list[str] = list(reversed(valid_dims))

    # Remove dims that have already been fixed.
    # Dim lists should be sort, so linear overhead is ok.
    for given in given_dims:
        if given is None:
            continue

        if given not in valid_dims:
            raise ValueError(
                f"invalid dimension '{given}' - must be one of {valid_dims}"
            )
        try:
            unused_dims.remove(given)
        except ValueError:
            raise ValueError(f"dimension '{given}' used more than once")

    resolved_dims: list[str] = []

    # Resolve dims by filling in 'None' values with unused dims.
    for given in given_dims:
        if given is None:
            resolved_dims.append(unused_dims.pop())
        else:
            resolved_dims.append(given)

    return resolved_dims
