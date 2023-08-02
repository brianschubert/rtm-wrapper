from __future__ import annotations

import abc
import dataclasses
from typing import Any, Callable, ClassVar

from typing_extensions import TypeAlias

from rtm_wrapper.parameters import Parameter
from rtm_wrapper.simulation import InputParameterName, Inputs, Outputs

ParameterHandler: TypeAlias = Callable[[...], None]


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
        for field in dataclasses.fields(inputs):
            param_name = field.name
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
    param_implementations: dict[
        tuple[InputParameterName, type[Parameter]], ParameterHandler
    ]

    def __init__(self) -> None:
        self.param_implementations = {}

    def register(self, name: InputParameterName, type_: type[Parameter]):
        def _register(func: ParameterHandler) -> ParameterHandler:
            self.param_implementations[(name, type_)] = func
            return func

        return _register
