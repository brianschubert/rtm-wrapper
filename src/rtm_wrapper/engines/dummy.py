"""
Mock RTM engines. Useful for development.
"""

from __future__ import annotations

import logging

from rtm_wrapper.engines.base import EngineOutputs, RTMEngine
from rtm_wrapper.simulation import Inputs


# TODO: update to new ouptut api
class DummyEngine(RTMEngine):
    def run_simulation(self, inputs: Inputs) -> EngineOutputs:
        logger = logging.getLogger(__name__)
        logger.info("%r", inputs)
        raise NotImplementedError


class NotImplementedEngine(RTMEngine):
    def run_simulation(self, inputs: Inputs) -> EngineOutputs:
        raise NotImplementedError
