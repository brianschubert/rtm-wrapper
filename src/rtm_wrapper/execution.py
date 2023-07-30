from __future__ import annotations

import abc
import concurrent.futures
import logging
import typing

import numpy as np
import xarray as xr

from rtm_wrapper.engines.base import RTMEngine
from rtm_wrapper.simulation import Outputs, SweepSimulation


class SweepExecutor(abc.ABC):
    """Base class for simulation executors."""

    @abc.abstractmethod
    def run(self, inputs: SweepSimulation, engine: RTMEngine):
        ...

    @abc.abstractmethod
    def collect_results(self) -> xr.Dataset:
        ...


class SerialExecutor(SweepExecutor):
    """Executor that runs simulations in series."""

    _results: xr.Dataset | None

    def __init__(self) -> None:
        self._results = None

    def run(self, sweep: SweepSimulation, engine: RTMEngine):
        self._results = xr.Dataset(coords=sweep.sweep_grid.coords)
        self._results = self._results.assign(
            {
                output_name: (
                    sweep.sweep_grid.dims,
                    np.empty(sweep.sweep_shape, dtype=output_type),
                )
                for output_name, output_type in typing.get_type_hints(Outputs).items()
            }
        )

        with np.nditer(sweep.sweep_grid.data, flags=["multi_index", "refs_ok"]) as it:
            for inputs in it:
                out = engine.run_simulation(inputs.item())
                self._results.variables["apparent_radiance"][
                    it.multi_index
                ] = out.apparent_radiance

    def collect_results(self) -> xr.Dataset:
        return self._results


class ConcurrentExecutor(SweepExecutor):
    """
    Executor the launches simulations in worker threads.

    This executor is designed to take advantage of engines that release the GIL
    while running.

    Assumes that the engine's ``run_simulation`` method is thread-safe.
    """

    _results: xr.Dataset | None
    _max_workers: int | None

    def __init__(self, max_workers: int | None = None) -> None:
        self._results = None
        self._max_workers = max_workers

    def run(self, sweep: SweepSimulation, engine: RTMEngine):
        logger = logging.getLogger(__name__)
        self._results = xr.Dataset(coords=sweep.sweep_grid.coords)
        self._results = self._results.assign(
            {
                output_name: (
                    sweep.sweep_grid.dims,
                    np.empty(sweep.sweep_shape, dtype=output_type),
                )
                for output_name, output_type in typing.get_type_hints(Outputs).items()
            }
        )

        # Execute simulations in worker threads.
        # This is fast so long as the engine release the GIL while running.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            # TODO add copy-engine interface for engines that are not thread safe
            futures_to_index = {
                executor.submit(
                    lambda idx: engine.run_simulation(sweep.sweep_grid.data[idx]),
                    idx,
                ): idx
                for idx in np.ndindex(sweep.sweep_shape)
            }

            for future in concurrent.futures.as_completed(futures_to_index):
                idx = futures_to_index[future]
                try:
                    out = future.result()
                    self._results.variables["apparent_radiance"][
                        idx
                    ] = out.apparent_radiance
                except Exception as ex:
                    error_input = sweep.sweep_grid.data[idx]
                    logger.error(
                        "exception occured when running simulation with input=%r",
                        error_input,
                        exc_info=ex,
                    )

    def collect_results(self) -> xr.Dataset:
        return self._results
