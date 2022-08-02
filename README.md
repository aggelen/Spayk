<div id="top"></div>

<!-- PROJECT SHIELDS -->
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SPAYK: An environment for spiking neural network simulation</h3>

  <p align="center">
    SPAYK is an open source tool developed to simulate spiking neurons and their networks. Documentation studies are still in progress.
    <br />
    <a href="https://github.com/aggelen/Spayk/tree/master/experiments"><strong>Experiments</strong></a>
    <br />
    <br />
    ·
    <a href="https://github.com/aggelen/spayk/issues">Issues</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Random Neurons with Izhikevich Model
100 Neurons: All Regular Spiking, Random const. current
![random_regular](https://github.com/aggelen/spayk/blob/master/static/random_regular.png)

100 Neurons: Random Dynamics with const. current 10mA
![random_all](https://github.com/aggelen/spayk/blob/master/static/random_dynamics.jpg)

### Synaptic Channel Models
Synaptic channel models for AMPA, NMDA, GABA_A and GABA_B are now available. Fig 3.2 from Neuronal Dynamics book can be re-created with example script at examples/synaptic_channels.py
![random_all](https://github.com/aggelen/spayk/blob/master/static/synaptic_channels.png)

### First Network
100 presynaptic neurons connected to 1 postsynaptic neuron with random weights. See examples/random100_neurons_to_1.py
![random_all](https://github.com/aggelen/spayk/blob/master/static/random100_to_1.png)

### Learning Attempt: STDP
STDP support now available. Please see [STDP Page](https://github.com/aggelen/Spayk/wiki/STDP) at Spayk Wiki.
![random_all](https://github.com/aggelen/spayk/blob/master/static/stdp/scn01_dW.png)

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started
SPAYK prerequisites and installation here!

### Prerequisites
See `requirements.txt` for more information.


### Installation
Publishing SPAYK as an installable package continues. Currently, you can directly download the repo and run it on your machine.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the BSD-3-Clause license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
