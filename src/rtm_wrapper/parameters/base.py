"""
Base classes defining the parameter tree.
"""
from __future__ import annotations

import abc
import contextlib
import copy
import re
from typing import Any, ClassVar, Generator, Generic, Mapping, TypeVar, Union, overload

import numpy as np
from typing_extensions import Literal, Self, TypeAlias

from .util import MetadataDict, ParameterError, UnsetParameterError

ParameterPath: TypeAlias = Union[str, tuple[str, ...]]
T = TypeVar("T")
F = TypeVar("F")

P = TypeVar("P", bound="Parameter")


_VALIDATE_FIELDS: bool = True


class Field(Generic[F]):
    """
    Base class for field descriptors.

    Fields are leaves in the input parameter tree that are responsible for storing
    fixed input parameter values.

    Each instance of a field may optionally specify a ``title`` and ``unit``, which
    help document  the meaning of the field and can be used in human-readable
    representations.
    """

    public_name: str
    """Name of this descriptor in the host class."""

    private_name: str
    """Attribute used to store this field's value in the host instance."""

    title: str | None
    """Human readable title of this field."""

    unit: str | None
    """Unit that this field is measured in."""

    dtype: ClassVar[np.dtype[Any]]
    """Numpy dtype used to store sweeps of this field."""

    def __init__(self, title: str | None = None, unit: str | None = None):
        self.title = title
        self.unit = unit

    def __set_name__(self, owner: Any, name: str) -> None:
        self.public_name = name
        self.private_name = f"_{name}"

    def __get__(self, instance: T, owner: type[T] | None) -> F:
        try:
            value = getattr(instance, self.private_name)
        except AttributeError:
            raise UnsetParameterError(
                f"attempted to access unset parameter"
                f" {type(instance).__name__}.{self.public_name}"
            )

        return value

    def __set__(self, instance: Any, value: F) -> None:
        if _VALIDATE_FIELDS:
            self.validate(instance, value)
        setattr(instance, self.private_name, value)

    def validate(self, instance: Any, value: F) -> None:
        pass  # TODO settle on validation framework

    def metadata(self) -> MetadataDict:
        metadata: MetadataDict = {}
        if self.title is not None:
            metadata["title"] = self.title
        if self.unit is not None:
            metadata["unit"] = self.unit
        return metadata


class ParameterMeta(type):
    """Metaclass for parameters."""

    _fields: frozenset[str]

    def __new__(
        cls: type[ParameterMeta],
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        /,
        **kwargs: Any,
    ) -> type:
        class_obj = super().__new__(cls, name, bases, namespace, **kwargs)

        fields = set()

        # TODO check for conflicts among base classes and current class?
        for base in bases:
            if not issubclass(type(base), ParameterMeta):
                continue
            fields.update(base._fields)  # type: ignore

        fields.update(
            name for name, value in namespace.items() if isinstance(value, Field)
        )

        class_obj._fields = frozenset(fields)

        return class_obj


class Parameter(metaclass=ParameterMeta):
    """
    Base class for input parameters.

    Parameter subclasses should represent some definite physical representation
    of a model parameter that RTM engines can optionally implement.
    """

    _fields: ClassVar[frozenset[str]]

    def __init__(self, **kwargs: Any) -> None:
        for name, value in kwargs.items():
            if name in self._fields:
                setattr(self, name, value)
            else:
                raise ParameterError(
                    f"unknown field '{name}' - must be one of {list(self._fields)}"
                )

    def __repr__(self) -> str:
        field_parts = []
        for name in sorted(self._fields):
            try:
                field_parts.append(f"{name}={getattr(self, name)!r}")
            except UnsetParameterError:
                field_parts.append(f"{name}=<UNSET>")

        return f"{type(self).__name__}({', '.join(field_parts)})"

    def replace(self, *args: Any, **kwargs: Any) -> Self:
        duplicate = copy.deepcopy(self)
        duplicate.set(*args, **kwargs)
        return duplicate

    @overload
    def set(self, param: ParameterPath, value: Any, /) -> None:
        ...

    @overload
    def set(self, updates: Mapping[str, Any], /) -> None:
        ...

    @overload
    def set(
        self,
        /,
        **kwargs: Any,
    ) -> None:
        ...

    def set(
        self,
        param: Mapping[str, Any] | ParameterPath | None = None,
        value: Any | None = None,
        /,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            if param is not None:
                raise ValueError(
                    "kwargs must not be passed when positional arguments are used"
                )
            for param_path, param_arg in kwargs.items():
                self.set(param_path, param_arg)
            return

        if param is None:
            # No positional arguments were given.
            # Called no with arguments - do nothing.
            return

        if isinstance(param, Mapping):
            for param_path, param_arg in param.items():
                self.set(param_path, param_arg)
            return

        if value is None:
            raise ParameterError(
                "value must be specified when first argument is not a mapping"
            )

        try:
            self._set(_parse_parameter_path(param), value)
        except Exception as ex:
            raise ParameterError(
                f"failed to set field '{param}' to '{value}': {ex}"
            ) from ex

    def _set(self, path: tuple[str, ...], value: Any) -> None:
        curr_field, *sub_path = path
        if sub_path:
            try:
                sub_param = getattr(self, curr_field)
            except AttributeError:
                raise ParameterError(f"unknown parameter '{curr_field}")
            sub_param._set(sub_path, value)
        else:
            # Not: can't use hasattr check, since it would invoke __get__ on <UNSET>
            # fields.
            if curr_field not in vars(self.__class__):
                raise ParameterError(f"unknown field '{curr_field}'")
            setattr(self, curr_field, value)

    @overload
    def get_fields(self, style: Literal[".", "__"] = ...) -> list[str]:
        ...

    @overload
    def get_fields(self, style: Literal["()"]) -> list[tuple[str, ...]]:
        ...

    def get_fields(
        self, style: Literal[".", "__", "()"] = "."
    ) -> list[str] | list[tuple[str, ...]]:
        """Return list containing the paths to all this parameter's terminal fields."""
        if style not in (".", "__", "()"):
            raise ValueError(f"unknown parameter path style '{style}'")

        paths = []
        for field_name in self._fields:
            field = vars(self.__class__)[field_name]
            if isinstance(field, ParameterField):
                for p in getattr(self, field_name).get_fields("()"):
                    paths.append((field_name,) + p)
            else:
                paths.append((field_name,))

        if style != "()":
            paths = [style.join(p) for p in paths]
        return paths

    def get_metadata(self, param: ParameterPath) -> MetadataDict:
        return self._get_metadata(_parse_parameter_path(param))

    def _get_metadata(self, path: tuple[str, ...]) -> MetadataDict:
        curr_field, *sub_path = path

        if sub_path:
            try:
                sub_param = getattr(self, curr_field)
            except AttributeError:
                raise ParameterError(f"unknown parameter '{curr_field}")
            return sub_param._get_metadata(sub_path)
        else:
            try:
                return vars(self.__class__)[curr_field].metadata()
            except KeyError:
                raise ParameterError(f"unknown field '{curr_field}'")


class AbstractParameterMeta(ParameterMeta, abc.ABCMeta):
    """Metaclass for abstract parameters."""


class AbstractParameter(Parameter, metaclass=AbstractParameterMeta):
    """
    Base class for abstract input parameters.

    Like ``Parameter``, but supports ``abc.ABC`` astract behavior.
    """


def _parse_parameter_path(param_path: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(param_path, tuple):
        return param_path

    return tuple(re.split(r"\.|__", param_path))


class ParameterField(Field[P]):
    """
    Field containing a swappable parameter.

    Creates a branch in the parameter tree.
    """

    _parameter_type: type[P]

    def __init__(self, parameter_type: type[P], title: str | None = None) -> None:
        super().__init__(title=title, unit=None)
        self._parameter_type = parameter_type

    def validate(self, instance: Any, value: F) -> None:
        if not isinstance(value, self._parameter_type):
            raise ParameterError(
                f"value for {type(instance).__name__}.{self.public_name}"
                f" must be subclass of {self._parameter_type.__name__},"
                f" got type {type(value).__name__}"
            )


class StrField(Field[str]):
    """Field taking on a string value."""

    dtype = np.dtype(str)


class FloatField(Field[float]):
    """Field taking on a float value."""

    dtype = np.dtype(float)


class IntField(Field[int]):
    """Field taking on an integer value."""

    dtype = np.dtype(int)


class FloatArrayField(Field[np.ndarray[Any, Any]]):
    """Field taking on a float value."""

    # TODO array validation
    dtype = np.dtype(float)


@contextlib.contextmanager
def validate_fields(flag: bool) -> Generator[None, None, None]:
    """Context manager for temporary enabling or disabling field validation."""
    global _VALIDATE_FIELDS
    prior = _VALIDATE_FIELDS
    try:
        _VALIDATE_FIELDS = flag
        yield
    finally:
        _VALIDATE_FIELDS = prior
