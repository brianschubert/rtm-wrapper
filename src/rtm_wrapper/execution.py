from __future__ import annotations

import abc
import base64
import concurrent.futures
import datetime
import gzip
import itertools
import logging
import math
import multiprocessing
import operator
import pathlib
import pickle
import tempfile
from abc import ABC
from typing import Any, Callable, Iterable, Literal

import numpy as np
import xarray as xr
from typing_extensions import Never

import rtm_wrapper.util as rtm_util
from rtm_wrapper.engines.base import EngineOutputs, OutputName, RTMEngine
from rtm_wrapper.simulation import SweepSimulation


class SweepExecutor(abc.ABC):
    """Base class for simulation executors."""

    @abc.abstractmethod
    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
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
        **kwargs: Any,
    ) -> None:
        self._allocate_results_like(sweep.sweep_spec, engine)
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
        self,
        sweep_spec: xr.Dataset,
        engine: RTMEngine,
    ) -> None:
        data_vars = {}

        sweep_dims = sweep_spec.indexes.dims

        for output_name in engine.requested_outputs:
            output_type = engine.outputs._dtypes[output_name]
            output_metadata = engine.outputs._metadata[output_name]

            if output_type == np.dtype(object):
                raise RuntimeError(
                    f"cannot sweep with output '{output_name}' of object type"
                )

            # TODO: tidy add mechanism to report dtypes for virtual outputs?
            if output_type is Never:
                raise RuntimeError(
                    f"sweeping with virtual output ('{output_name}') not yet supported. "
                    f"As a quick-fix, register an extracted output that mirrors it."
                )

            data_vars[output_name] = (
                # All output variables have the same shape as the input grid.
                tuple(sweep_dims.keys()),
                np.empty(tuple(sweep_dims.values()), dtype=output_type),
                # Add output metadata as attributes.
                output_metadata,
            )

        self._results = xr.Dataset(data_vars, coords=sweep_spec.coords)


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

        for idx in np.ndindex(sweep.sweep_shape):
            inputs = sweep[idx]
            out = engine.run_simulation(inputs)  # type: ignore
            for output_name in self._results.keys():
                self._results.data_vars[output_name][idx] = out[output_name]  # type: ignore
            if step_callback is not None:
                step_callback(idx)


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

    _max_workers: int

    def __init__(
        self,
        max_workers: int | None = None,
    ) -> None:
        super().__init__()
        if max_workers is None:
            # Override ThreadPoolExecutor's default.
            # Engines tend to be CPU bound (not I/O bound), so the default of more
            # threads than cores is detrimental.
            max_workers = multiprocessing.cpu_count()
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

            def target(idx: tuple[int, ...]) -> EngineOutputs:
                return engine.run_simulation(sweep[idx])  # type: ignore

            futures_to_index = {
                executor.submit(target, idx): idx
                for idx in np.ndindex(sweep.sweep_shape)
            }

            for future in concurrent.futures.as_completed(futures_to_index):
                idx = futures_to_index[future]
                try:
                    out = future.result()
                    for output_name in self._results.keys():
                        self._results.variables[output_name][idx] = out[output_name]  # type: ignore
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


class ParallelConcurrentExecutor(SweepExecutor):
    """
    Executor that runs multiple ``ConcurrentExecutor``s in spawned subprocess.

    This can improve performance over ``ConcurrentExecutor`` when simulation sweeps
    are Python bounded, which can happen when individual simulator runs are fast or
    when many simulation works are used.
    """

    _split_dim: str | None = None

    _result_files: list[pathlib.Path]

    _max_managers: int
    _max_workers: int

    def __init__(
        self,
        split_dim: str | None = None,
        max_managers: int | None = None,
        max_workers: int | None = None,
    ) -> None:
        self._split_dim = split_dim
        self._result_files = []
        self._max_managers = max_managers
        self._max_workers = max_workers

    def run(
        self,
        sweep: SweepSimulation,
        engine: RTMEngine,
        *,
        work_directory: pathlib.Path | None = None,
        step_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._split_dim is None:
            self._split_dim = max(sweep.dims.items(), key=operator.itemgetter(1))[0]

        # TODO consider using lazy splitting
        split_sweeps = list(sweep.split(sweep.dims[self._split_dim], self._split_dim))

        if work_directory is not None:
            self._run_sims(split_sweeps, engine, work_directory, step_callback)
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                work_directory = pathlib.Path(temp_dir)
                self._run_sims(split_sweeps, engine, work_directory, step_callback)

    def _run_sims(
        self,
        sweeps: list[SweepSimulation],
        engine: RTMEngine,
        work_directory: pathlib.Path,
        step_callback: Callable[[str], None] | None = None,
    ) -> None:
        file_prefix = datetime.datetime.now().astimezone().isoformat()
        ndigits = math.floor(math.log10(len(sweeps)) + 1)
        file_gen = (
            work_directory.joinpath(f"{file_prefix}_{i:0{ndigits}d}.nc")
            for i in itertools.count()
        )

        work_directory.mkdir(exist_ok=True)

        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self._max_managers,
            mp_context=context,
        ) as executor:
            futures_to_file = {
                executor.submit(
                    _parallel_sim_target,
                    swp,
                    engine,
                    save_file,
                    self._max_workers,
                ): save_file
                for (swp, save_file) in zip(sweeps, file_gen)
            }
            self._result_files = list(futures_to_file.values())
            print("2")

            # TODO any handling?
            for future in concurrent.futures.as_completed(futures_to_file):
                save_file = futures_to_file[future]
                print(3)
                try:
                    future.result()
                except Exception:
                    raise
                if step_callback is not None:
                    step_callback(save_file)

    def collect_results(self) -> xr.Dataset:
        if not self._result_files:
            raise RuntimeError("no results to load")

        ds = [xr.load_dataset(file) for file in self._result_files]
        print(ds)
        # TODO assess performance vs xr.open_mfdataset
        # TODO dataset attribtues
        results = xr.combine_nested(
            ds,
            concat_dim=self._split_dim,
            compat="equals",
            join="exact",
            combine_attrs="drop",
        )
        return results


def _parallel_sim_target(
    sweep: SweepSimulation, engine: RTMEngine, save_file: pathlib.Path, max_workers: int
) -> None:
    executor = ConcurrentExecutor(max_workers=max_workers)
    executor.run(sweep, engine)
    results = executor.collect_results()
    results.to_netcdf(save_file)
