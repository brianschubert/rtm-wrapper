import abc

from rtm_wrapper.simulation import Inputs, Outputs


class RTMEngine(abc.ABC):
    """
    Base class for wrappers interfaces around specific RTMs.
    """

    @abc.abstractmethod
    def run_simulation(self, inputs: Inputs) -> Outputs:
        ...
