"""
Mock RTM engines. Useful for development.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Final

from rtm_wrapper.engines.base import RTMEngine
from rtm_wrapper.simulation import Inputs, Outputs

_NAN_OUTPUTS: Final = Outputs(
    **{f.name: float("nan") for f in dataclasses.fields(Outputs)}
)


class DummyEngine(RTMEngine):
    def run_simulation(self, inputs: Inputs) -> Outputs:
        logger = logging.getLogger(__name__)
        logger.info("%r", inputs)
        return dataclasses.replace(_NAN_OUTPUTS)


class NotImplementedEngine(RTMEngine):
    def run_simulation(self, inputs: Inputs) -> Outputs:
        raise NotImplementedError
