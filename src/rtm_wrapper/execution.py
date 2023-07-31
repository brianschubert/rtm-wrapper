from __future__ import annotations

import abc
import concurrent.futures
import logging
import typing
from abc import ABC
from typing import Callable

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


class LocalMemoryExecutor(SweepExecutor, ABC):
    """
    Base class for executors that store the entire simulation results in a local,
    in-memory ``xarray.Dataset``.
    """

    _results: xr.Dataset | None

    def __init__(self) -> None:
        self._results = None

    def collect_results(self) -> xr.Dataset:
        return self._results

    def _allocate_results_like(self, sweep_grid: xr.DataArray) -> None:
        self._results = xr.Dataset(
            {
                # Preallocate variables for each output.
                output_name: (
                    # All output variables have the same shape as the input grid.
                    sweep_grid.dims,
                    np.empty(sweep_grid.data.shape, dtype=output_type),
                )
                for output_name, output_type in typing.get_type_hints(Outputs).items()
            },
            coords=sweep_grid.coords,
        )


class SerialExecutor(LocalMemoryExecutor):
    """Executor that runs simulations in series."""

    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        step_callback: Callable[[tuple[int, ...]], None] | None = None,
    ):
        self._allocate_results_like(sweep.sweep_grid)

        with np.nditer(sweep.sweep_grid.data, flags=["multi_index", "refs_ok"]) as it:
            for inputs in it:
                out = engine.run_simulation(inputs.item())
                for output_name in typing.get_type_hints(Outputs):
                    self._results.variables[output_name][it.multi_index] = getattr(
                        out, output_name
                    )
                if step_callback is not None:
                    step_callback(it.multi_index)


class ConcurrentExecutor(LocalMemoryExecutor):
    """
    Executor the launches simulations in concurrent worker threads.

    This executor is designed to take advantage of engines that release the GIL
    while running.

    **WARNING**: this executor assumes that the provided engine's ``run_simulation``
    method is thread-safe. All worker threads operate on the same engine instance.
    Make sure that the provided engine *does not* mutate itself or any global state
    without appropriate locking.
    """

    _max_workers: int | None

    def __init__(
        self,
        max_workers: int | None = None,
    ) -> None:
        super().__init__()
        self._max_workers = max_workers

    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        step_callback: Callable[[tuple[int, ...]], None] | None = None,
    ):
        logger = logging.getLogger(__name__)
        self._allocate_results_like(sweep.sweep_grid)

        # Execute simulations in worker threads.
        # This is fast so long as the engine releases the GIL while running.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
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
                    for output_name in typing.get_type_hints(Outputs):
                        self._results.variables[output_name][idx] = getattr(
                            out, output_name
                        )
                except Exception as ex:
                    error_input = sweep.sweep_grid.data[idx]
                    logger.error(
                        "exception occurred when running simulation with input=%r, idx=%r",
                        error_input,
                        idx,
                        exc_info=ex,
                    )
                if step_callback is not None:
                    step_callback(idx)
