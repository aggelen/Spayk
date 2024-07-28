Spike Generation
============

Spike Trains
---------------

In Spayk, each spike data is formatted as a spike train. You can generate a Poisson Spike Train to work on.::

    from spayk.Stimuli import PoissonSpikeTrain
    spike_train = PoissonSpikeTrain(3, np.array([15,25,35]), [0, 1, 1e-3])

This indicates that you want to generate a Poisson spike train for three neurons. It is produced at a firing rate of 15,25,35 Hz for each neuron, respectively. The last parameter is a list representing the start time, end time and delta time respectively in seconds.

Now, you can use statistics and plot functions of spike train class.::

    spike_train.raster_plot()
    spike_train.firing_rates()

These classes use the "poisson_spike_generator" function in spayk.Data to generate spikes from the Poisson distribution. You can use directly this function to create spike data as an ndarray
This example generates spike data for 3 neurons with 5,10,15 Hz firing rates respectively for each neuron. The last parameter is a list that controls simulation time.
Total spike generation process starts at time 0 to time 1 with delta time of 1e-3, each in seconds.::

    spike_data = poisson_spike_generator(3, np.array([5,10,15]), [0, 1, 1e-3])
