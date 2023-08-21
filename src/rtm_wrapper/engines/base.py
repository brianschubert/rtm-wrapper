from __future__ import annotations

import abc
import dataclasses
import inspect
import typing
from typing import Any, Callable, ClassVar

from typing_extensions import Never, TypeAlias

import rtm_wrapper.util as rtm_util
from rtm_wrapper.parameters import Parameter
from rtm_wrapper.simulation import INPUT_TOP_NAMES, Inputs, InputTopName, Outputs

ParameterHandler: TypeAlias = Callable[..., None]


class RTMEngine(abc.ABC):
    """
    Base class for wrappers interfaces around specific RTMs.
    """

    params: ClassVar[ParameterRegistry]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.params = ParameterRegistry()

    @abc.abstractmethod
    def run_simulation(self, inputs: Inputs) -> Outputs:
        ...

    def load_inputs(
        self, inputs: Inputs, *handler_args: Any, **handlers_kwargs: Any
    ) -> None:
        for param_name in INPUT_TOP_NAMES:
            param_value = getattr(inputs, param_name)
            try:
                handler = self.__class__.params.param_implementations[
                    (param_name, type(param_value))
                ]
            except KeyError as ex:
                raise RuntimeError(
                    f"engine {self.__class__.__name__} has not implemented parameter "
                    f"type {type(param_value).__name__} for input '{param_name}'"
                ) from ex
            handler(param_value, *handler_args, **handlers_kwargs)


class ParameterRegistry:
    param_implementations: dict[tuple[InputTopName, type[Parameter]], ParameterHandler]

    def __init__(self) -> None:
        self.param_implementations = {}

    def register(
        self, name: InputTopName, type_: type[Parameter] | None = None
    ) -> Callable[[ParameterHandler], Callable[..., Never]]:
        def _register(func: ParameterHandler) -> Callable[..., Never]:
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

            self.param_implementations[(name, param_type)] = func
            return _dont_call_parameter_handler

        return _register


def _dont_call_parameter_handler(*_args: Any, **_kwargs: Any) -> Never:
    raise RuntimeError(
        "parameter handler should not be called directly. Access the handler "
        "through corresponding engine's ParameterRegistry"
    )
