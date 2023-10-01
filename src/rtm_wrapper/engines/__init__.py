"""
RTM-specific engines.

Engines are wrappers around a specific RTM. They are responsible for

1. translating  :class:`rtm_wrapper.simulation.Inputs` trees into the configuration for
   a specific RTM,
2. running an RTM simulation, and
3. extracting desired outputs from the simulation results.

The majority of an engines behaves is defined by its
:attr:`rtm_wrapper.engines.RTMEngine.params` and
:attr:`rtm_wrapper.engines.RTMEngine.outputs` class variables.
These objects contain registries of callback functions that the engine will use to
implement the engine's support for various input parameters or output values.
"""
