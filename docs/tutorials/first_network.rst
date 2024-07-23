First Network
============

Neural Circuit
---------------

In Spayk, there is a new class, ie. NeuralCircuit::

    neurons = None
    synapses = None
    nc = NeuralCircuit(neurons, synapses)

The neural circuit is a problem generator for any simple neuron group with synapses. It generates a executable dynamic code for the simulation.



Synapse Formation
----------------

Note that this produces the following structure of excitatory connections:

       | from E1  from E2  from E3
 ---------------------------------
  to E1 |   w      w       w
  to E2 |   w      w       w
  to E3 |   w      w       w