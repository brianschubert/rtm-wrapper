"""
Mock RTM engines. Useful for development.
"""

from __future__ import annotations

import logging

from typing_extensions import Never

from rtm_wrapper.engines.base import EngineOutputs, RTMEngine
from rtm_wrapper.simulation import Inputs


# TODO: update to new ouptut api
class DummyEngine(RTMEngine):
    """Dummy engine that logs its inputs and produces no outputs."""

    def run_simulation(self, inputs: Inputs) -> EngineOutputs:
        logger = logging.getLogger(__name__)
        logger.info("%r", inputs)
        return {}


class NotImplementedEngine(RTMEngine):
    """Dummy engine that raises ``NotImplementedError`` when run."""

    def run_simulation(self, inputs: Inputs) -> Never:
        raise NotImplementedError
