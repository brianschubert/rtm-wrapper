from __future__ import annotations

import abc
import base64
import concurrent.futures
import datetime
import gzip
import logging
import pickle
import typing
from abc import ABC
from typing import Any, Callable, Iterable, Literal

import numpy as np
import xarray as xr

import rtm_wrapper.util as rtm_util
from rtm_wrapper.engines.base import RTMEngine
from rtm_wrapper.simulation import OUTPUT_NAMES, OutputName, Outputs, SweepSimulation


class SweepExecutor(abc.ABC):
    """Base class for simulation executors."""

    @abc.abstractmethod
    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        *,
        outputs: Iterable[OutputName] | None = None,
        **kwargs: Any,
    ) -> None:
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

    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        *,
        outputs: Iterable[OutputName] | None = None,
        **kwargs: Any,
    ) -> None:
        if outputs is not None:
            outputs = frozenset(outputs)
            extraneous_outputs = outputs - OUTPUT_NAMES
            if extraneous_outputs:
                raise ValueError(f"unknown output names {list(extraneous_outputs)}")
        else:
            outputs = OUTPUT_NAMES

        self._allocate_results_like(sweep.sweep_spec.grid, outputs)
        assert self._results is not None  # for type checker

        sim_start = datetime.datetime.now().astimezone().isoformat()
        self._run(sweep, engine, **kwargs)
        sim_end = datetime.datetime.now().astimezone().isoformat()

        engine_type = type(engine)
        # Populate metadata attributes
        self._results = self._results.assign_attrs(
            {
                "version": rtm_util.build_version(),
                "platform": rtm_util.platform_summary(),
                "engine": f"{engine_type.__module__}.{engine_type.__qualname__}",
                "base_repr": repr(sweep.base),
                "base_pzb64": base64.b64encode(
                    gzip.compress(pickle.dumps(sweep.base))
                ).decode(),
                "sim_start": sim_start,
                "sim_end": sim_end,
            }
        )

    @abc.abstractmethod
    def _run(self, sweep: SweepSimulation, engine: RTMEngine, **kwargs: Any) -> None:
        ...

    def collect_results(self) -> xr.Dataset:
        if self._results is None:
            raise ValueError("no simulations have been run yet")
        return self._results

    def _allocate_results_like(
        self, sweep_grid: xr.DataArray, variables: Iterable[OutputName]
    ) -> None:
        data_vars = {}

        # Preallocate variables for each requested output.
        outputs_hints = typing.get_type_hints(Outputs, include_extras=True)
        for output_name in variables:
            output_type, output_metadata = typing.get_args(outputs_hints[output_name])
            data_vars[output_name] = (
                # All output variables have the same shape as the input grid.
                sweep_grid.dims,
                np.empty(sweep_grid.data.shape, dtype=output_type),
                # Add output metadata as attributes.
                output_metadata,
            )

        self._results = xr.Dataset(data_vars, coords=sweep_grid.coords)


class SerialExecutor(LocalMemoryExecutor):
    """Executor that runs simulations in series."""

    def _run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        *,
        step_callback: Callable[[tuple[int, ...]], None] | None = None,
        **kwargs: Any,
    ) -> None:
        assert self._results is not None  # for type checker

        if kwargs:
            raise ValueError(f"unknown kwargs {kwargs}")

        with np.nditer(
            sweep.sweep_spec.grid.data, flags=["multi_index", "refs_ok"]
        ) as it:
            for inputs in it:
                out = engine.run_simulation(inputs.item())  # type: ignore
                for output_name in self._results.keys():
                    self._results.data_vars[output_name][it.multi_index] = getattr(
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

    def _run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        *,
        step_callback: Callable[[tuple[int, ...]], None] | None = None,
        on_error: Literal["ignore", "abort"] = "abort",
        **kwargs: Any,
    ) -> None:
        assert self._results is not None  # for type checker

        if kwargs:
            raise ValueError(f"unknown kwargs {kwargs}")

        logger = logging.getLogger(__name__)

        # Execute simulations in worker threads.
        # This is fast so long as the engine releases the GIL while running.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            futures_to_index = {
                executor.submit(
                    lambda idx: engine.run_simulation(sweep[idx]),
                    idx,
                ): idx
                for idx in np.ndindex(sweep.sweep_shape)
            }

            for future in concurrent.futures.as_completed(futures_to_index):
                idx = futures_to_index[future]
                try:
                    out = future.result()
                    for output_name in self._results.keys():
                        self._results.variables[output_name][idx] = getattr(
                            out, output_name
                        )
                except Exception as ex:
                    error_input = sweep[idx]
                    logger.error(
                        "exception occurred when running simulation with input=%r, idx=%r",
                        error_input,
                        idx,
                        exc_info=ex,
                    )
                    if on_error == "abort":
                        raise
                if step_callback is not None:
                    step_callback(idx)
