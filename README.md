<div id="top"></div>

<div align="center">

  <a href="https://github.com/aggelen/Spayk/stargazers">![Stargazers](https://img.shields.io/github/stars/aggelen/Spayk.svg?style=for-the-badge)</a>
  <a href="https://github.com/aggelen/Spayk/issues">![Issues](https://img.shields.io/github/issues/aggelen/Spayk.svg?style=for-the-badge)</a>
  <a href="https://github.com/aggelen/Spayk/blob/master/LICENSE">![license](https://img.shields.io/github/license/aggelen/Spayk.svg?style=for-the-badge)</a>

</div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <h3 align="center">SPAYK: An environment for spiking neural network simulation</h3>

  <p align="center">
    SPAYK is an open source tool developed to simulate spiking neurons and their networks. Documentation studies are still in progress.
    <br />
    <a href="https://github.com/aggelen/Spayk/tree/master/experiments"><strong>Experiments</strong></a>
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
We recommend creating a new environment to use Spayk.

```
conda create --name spayk python==3.8 matplotlib==3.5.0 numpy==1.21.2 seaborn==0.11.2 tqdm==4.62.3
```

After creating the environment, you can activate it and run the examples.

```
conda activate spayk

```

### Prerequisites
See `requirements.txt` for more information.

Spayk uses torchvision for MNIST experiments. Before you begin, please install it.

```
pip install torchvision
```


### Installation
```
pip install Spayk
```

Please see project at: https://pypi.org/project/Spayk/

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the BSD-3-Clause license. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>


