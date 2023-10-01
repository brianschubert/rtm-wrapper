"""
RTM engine definition.
"""
from __future__ import annotations

import abc
import graphlib
import inspect
import typing
from collections.abc import Iterable
from typing import Any, Callable, ClassVar, Generic, TypeVar

import numpy as np
from typing_extensions import Concatenate, Never, ParamSpec, TypeAlias

import rtm_wrapper.util as rtm_util
from rtm_wrapper.parameters import MetadataDict, Parameter
from rtm_wrapper.simulation import INPUT_TOP_NAMES, Inputs, InputTopName

# Trailing parameter specification of decorated function.
P = ParamSpec("P")
R = TypeVar("R", bound=Parameter)

OutputName: TypeAlias = str
Output: TypeAlias = Any

# Decorator targets.
ParameterHandler: TypeAlias = Callable[Concatenate[R, P], None]

EngineOutputs: TypeAlias = dict[OutputName, Output]
OutputExtractor: TypeAlias = Callable[..., Output]


class RTMEngine(abc.ABC):
    """
    Base class for wrappers interfaces around specific RTMs.
    """

    params: ClassVar[ParameterRegistry[...]]
    """
    Registry of the input parameters that this engine can handle.
    """

    outputs: ClassVar[OutputRegistry]
    """
    Registry of the outputs that can be extracted from run of this engine.
    """

    virtual_outputs: ClassVar[tuple[OutputName, ...]] = ()
    """
    Mock outputs produced by this engine without invoking an output extractor.
    """

    default_outputs: ClassVar[tuple[OutputName, ...]]
    """
    The default outputs returned from runs of this engine.
    
    This can be overridem by passing the ``outputs`` keyword argument to ``__init__``.
    """

    _requested_outputs: tuple[OutputName, ...]

    _extraction_order: tuple[OutputName, ...]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.params = ParameterRegistry()
        cls.outputs = OutputRegistry()

        # Pre-register all virtual outputs so that actual output extractors can use
        # them as dependencies.
        for output_name in cls.virtual_outputs:
            cls.outputs.register(
                rtm_util.trap("attempted to extract virtual output - this is a bug"),
                depends=(),
                name=output_name,
            )

    def __init__(self, *, outputs: Iterable[OutputName] | None = None) -> None:
        if outputs is None:
            outputs = self.default_outputs

        self.requested_outputs = tuple(outputs)

    @property
    def requested_outputs(self) -> tuple[OutputName, ...]:
        """
        Currently configured outputs for this engine.

        Re-assign to change the configured outputs.
        """
        return self._requested_outputs

    @requested_outputs.setter
    def requested_outputs(self, outputs: tuple[OutputName, ...]) -> None:
        not_implemented = frozenset(outputs).difference(self.outputs.names)
        if not_implemented:
            raise ValueError(
                f"unknown outputs {list(not_implemented)}."
                f" See '{self.__class__.__name__}.outputs.names' for available outputs."
            )

        self._requested_outputs = outputs

        # Precompute the output extractions once so that we don't incur a penalty on
        # each engine run.
        self._extraction_order = tuple(
            step
            for step in self.outputs.extraction_order(self._requested_outputs)
            if step not in self.virtual_outputs
        )

    @abc.abstractmethod
    def run_simulation(self, inputs: Inputs) -> EngineOutputs:
        """
        Run this RTM module using the given inputs.

        :param inputs: Input parameter tree.
        """

    def _extract_outputs(self, outputs: EngineOutputs) -> None:
        # TODO: verify that virtuals have been set?

        for output_name in self._extraction_order:
            dep_names, extractor = self.outputs._extractors[output_name]

            try:
                # TODO: is generator or generator->tuple measurably faster?
                extractor_args = [outputs[dep] for dep in dep_names]
            except KeyError as ex:
                raise RuntimeError(
                    f"output predecessor '{ex.args[0]}' for '{output_name}' not set"
                    f" - this is a bug. Extraction order was {self._extraction_order}"
                ) from ex

            outputs[output_name] = extractor(*extractor_args)


class ParameterRegistry(Generic[P]):
    """Registry of input parameters supported by an RTM engine."""

    param_implementations: dict[
        tuple[InputTopName, type[Parameter]], ParameterHandler[Parameter, P]
    ]

    def __init__(self) -> None:
        self.param_implementations = {}

    def register(
        self, name: InputTopName, type_: type[Parameter] | None = None
    ) -> Callable[[ParameterHandler[R, P]], ParameterHandler[R, P]]:
        """Return decorator for registering a new input parameter."""

        def _register(func: ParameterHandler[R, P]) -> Callable[..., Never]:
            if type_ is None:
                # Infer type from annotation of first positional argument.

                first_param = rtm_util.first_or(inspect.signature(func).parameters)
                if first_param is None:
                    raise ValueError(
                        "decorated function must accept at least one positional argument"
                    )

                hints = typing.get_type_hints(func)
                try:
                    param_type = hints[first_param]
                except KeyError:
                    raise ValueError(
                        "first argument of decorator function is missing an annotation"
                    )
            else:
                param_type = type_

            if not issubclass(param_type, Parameter):
                raise ValueError(
                    f"parameter must be a Parameter subclass, got {param_type}"
                )

            self.param_implementations[(name, param_type)] = typing.cast(
                ParameterHandler[Parameter, P], func
            )
            return rtm_util.trap(
                "parameter handler should not be called directly. "
                "Access the handler through corresponding engine's ParameterRegistry"
            )

        return _register

    def process(self, inputs: Inputs, *args: P.args, **kwargs: P.kwargs) -> None:
        for param_name in INPUT_TOP_NAMES:
            param_value = getattr(inputs, param_name)
            try:
                handler = self.param_implementations[(param_name, type(param_value))]
            except KeyError as ex:
                raise RuntimeError(
                    f"missing handler for type '{type(param_value).__name__}' as input '{param_name}'"
                ) from ex
            handler(param_value, *args, **kwargs)


class OutputRegistry:
    """Registry of outputs that can be extracted from an RTM engine run."""

    _extractors: dict[OutputName, tuple[tuple[OutputName, ...], OutputExtractor]]

    _metadata: dict[OutputName, MetadataDict]

    _dtypes: dict[OutputName, np.dtype[Any]]

    def __init__(self) -> None:
        self._extractors = {}
        self._metadata = {}
        self._dtypes = {}

    @property
    def names(self) -> Iterable[OutputName]:
        """Return an iterable of the names of all outputs that have been registered."""
        return self._extractors.keys()

    def register(
        self,
        func: OutputExtractor | None = None,
        /,
        name: str | None = None,
        depends: tuple[OutputName, ...] | None = None,
        title: str | None = None,
        unit: str | None = None,
        dtype: np.dtype[Any] | None = None,
    ) -> Callable[[OutputExtractor], OutputExtractor]:
        """
        Return a decorator for registering an RTM engine output.
        """

        def _register(func: OutputExtractor) -> OutputExtractor:
            output_name = name if name is not None else func.__name__

            if output_name in self.names:
                raise ValueError(f"output '{output_name}' already registered")

            if depends is not None:
                output_dependencies = depends
            else:
                output_dependencies = tuple(inspect.signature(func).parameters)

            missing = [pred for pred in output_dependencies if pred not in self.names]
            if missing:
                # TODO swap with custom exception
                # TODO: maybe remove available outputs
                raise RuntimeError(
                    f"unable to register output {output_name}"
                    f" - depends on unregistered  predecessors {missing}."
                    f" Available: {list(self.names)}"
                )

            self._extractors[output_name] = (output_dependencies, func)
            metadata = MetadataDict()
            if title is not None:
                metadata["title"] = title
            if unit is not None:
                metadata["unit"] = unit
            self._metadata[output_name] = metadata

            if dtype is not None:
                resolved_dtype = dtype
            else:
                hints = typing.get_type_hints(func)
                try:
                    resolved_dtype = hints["return"]
                except KeyError:
                    raise RuntimeError(
                        "must pass dtype or include return type hint in decorated output extractor"
                    )
            self._dtypes[output_name] = resolved_dtype

            return func

        if func is None:
            return _register
        else:
            _register(func)
            return rtm_util.trap(
                f"{self.__class__.__name__}.{self.__class__.register.__name__} "
                f"cannot be used as decorator when func is passed"
            )

    def extraction_order(
        self, requested: Iterable[OutputName]
    ) -> tuple[OutputName, ...]:
        """
        Return a minial static extraction order for extracting the requested outputs.

        :param requested: Names of outputs that must be extracted.
        :return: Valid extraction order for obtaining the requested outputs. Includes
                 the requested outputs and their minimal prerequisites.
        """
        processed = set()
        required_extractions = set(requested)
        pending = set(required_extractions)

        while pending:
            out = pending.pop()
            processed.add(out)

            deps = frozenset(self._extractors[out][0])
            required_extractions |= deps
            pending |= deps - processed

        graph_order = graphlib.TopologicalSorter(
            {output_name: deps for output_name, (deps, _) in self._extractors.items()}
        )

        return tuple(
            step for step in graph_order.static_order() if step in required_extractions
        )
