#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 21:32:58 2022

@author: aggelen
"""
import numpy as np
import matplotlib.pyplot as plt

# Principles of Computational Modelling in Neuroscience
# Sterratt, Graham, Gillies, Willshaw
# Cambridge University Press, June 2011
# Fig. 7.3: Synaptic conductance waveforms responding to trains of APs
# B. Graham, Computing Science & Maths, University of Stirling

tmax=100
dt=0.1
t = np.arange(0,tmax,dt)

sp = np.zeros(len(t))
sp[200] = 1
sp[400] = 1
sp[600] = 1
sp[800] = 1


tau = 10


z = np.zeros(len(t))

g2 = np.zeros(len(t))
fdt = np.exp(-dt/tau)

# for i=2:length(t)
#   z(i) = (z(i-1)*fdt) + (tau*(1-fdt)/dt)*sp(i-1);
#   g2(i) = (g2(i-1)*fdt) + (z(i-1)*tau*(1-fdt));
# end;

for i in range(1,len(t)):
    z[i] = (z[i-1]*fdt) + (tau*(1-fdt)/dt)*sp[i-1]
    g2[i] = (g2[i-1]*fdt) + (z[i-1]*tau*(1-fdt))

plt.plot(g2)

# % Approx 1
# % (as used by Graham & Redman, 1994 for force in a muscle fibre)

# g1=zeros(1,length(t));
# fdt=1-(dt/tau);
# for i=3:length(t)
#   g1(i) = (2*fdt*g1(i-1)) - ((fdt^2)*g1(i-2)) + (((dt/tau)^2)*sp(i-2));
# end;

# % Approx 2
# % (as used by Wilson & Bower, 1989 in GENESIS)
# z=zeros(1,length(t));
# g2=zeros(1,length(t));
# fdt=exp(-dt/tau);
# for i=2:length(t)
#   z(i) = (z(i-1)*fdt) + (tau*(1-fdt)/dt)*sp(i-1);
#   g2(i) = (g2(i-1)*fdt) + (z(i-1)*tau*(1-fdt));
# end;

# tsize=12;
# lsize=9;
# nsize=9;

# plot(t,g1./max(g1),'k-');
# xlabel('t (msecs)','FontSize',lsize,'FontName','Helvetica');
# ylabel('Conductance','FontSize',lsize,'FontName','Helvetica');
# axis([0 tmax 0 1.02]);

# set(gca,'Box','off');
# set(findobj('Type','line'),'LineWidth',0.8);
# set(findobj('Type','text'),'FontSize',nsize,'FontName','Helvetica');
