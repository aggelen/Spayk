#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:34:27 2024

@author: gelenag
"""

import matplotlib.pyplot as plt
import numpy as np


class ProblemGenerator:
    def __init__(self):
        # self.verbose = True
        self.code_string = """"""
        self.has_analyzed = False

    def analyze_network(self, neurons, synapses, stimulus):
        total_no_neurons = 0
        # for neuron_group_name, neuron_group in neurons.items():
        #     total_no_neurons += neuron_group.no_neurons

        self.has_analyzed = True

    def make_sysofodes(self, neurons, synapses, stimulus):
        if self.has_analyzed:
            self.code_string += """import numpy as np\n"""
            self.code_string += """class Problem:\n\tdef __init__(self):\n"""

            neurongroup_dict = {}
            
            self.code_string += "\t\tself.dt = 0.1e-3\n"

            # generate unique id for every neuron
            # generate neuron memb. potential vars
            self.code_string += "\t\t#%% neuron memb. potens.\n"
            for neuron_group in neurons:
                neurongroup_dict[neuron_group.group_label] = neuron_group

                for neuron_label in neuron_group.neuron_labels:
                    self.code_string += "\t\t# >>>>>>>>> for neuron {}\n".format(
                        neuron_label)
                    for state_label in neuron_group.state_vector:
                        self.code_string += "\t\tself.{}_{} = 0.0\n".format(
                            state_label, neuron_label)
                        # self.code_string += "\t\tself.d_{}_{} = 0.0\n".format(state_label, neuron_label)

            #%% generate derivative reset func
            self.code_string += "\t#%% derivative reset func\n"
            self.code_string += "\tdef dx_reset(self):\n"
            for neuron_group in neurons:
                neurongroup_dict[neuron_group.group_label] = neuron_group
                for neuron_label in neuron_group.neuron_labels:
                    for state_label in neuron_group.state_vector:
                        self.code_string += "\t\tself.d_{}_{} = 0.0\n".format(
                            state_label, neuron_label)
                        
            #%% generate euler integration func
            self.code_string += "\t#%% euler integration func\n"
            self.code_string += "\tdef integrate_all_euler(self):\n"
            for neuron_group in neurons:
                neurongroup_dict[neuron_group.group_label] = neuron_group
                for neuron_label in neuron_group.neuron_labels:
                    for state_label in neuron_group.state_vector:
                        self.code_string += "\t\tself.{}_{} += self.d_{}_{}*self.dt\n".format(
                            state_label, neuron_label,state_label, neuron_label)

            #%% generate derivative func
            self.code_string += "\t#%% derivative func\n"
            self.code_string += "\tdef calculate_dxdt_all(self, stimuli):\n"

            self.code_string += "\t\tself.dx_reset()\n"
            
            self.code_string += "\t\t##% memb pot derivatives\n"
            for neuron_group in neurons:
                for neuron_label in neuron_group.neuron_labels:
                    self.code_string += "\t\tself.d_{}_{} = (-{}*(self.{}_{} - {}) - self.Isyn_{}) / {}\n".format(neuron_group.state_vector[0], 
                                                                                                           neuron_label,
                                                                                                           neuron_group.params['GL'],
                                                                                                           neuron_group.state_vector[0],
                                                                                                           neuron_label,
                                                                                                           neuron_group.params['VL'],
                                                                                                           neuron_label,
                                                                                                           neuron_group.params['CM'])
            
            self.code_string += "\t\t##% channel derivatives\n"
            for synapse_group in synapses:
                for channel_id, channel in enumerate(synapse_group.channels):
                    channel_state = synapse_group.state_labels[channel_id]
                    # recurrent ampa
                    if channel == 'AMPA':
                        if '[' in synapse_group.target:
                            target_group = neurongroup_dict[synapse_group.target.split('[')[0]]
                            
                            self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"

                            assoc_neuron_labels = synapse_group.target.split('[')[
                                1][:-1].split(':')
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
                                target_group.neuron_channel_assoc_table[neuron_id, 0] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_AMPA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)
                        else:
                            target_group = neurongroup_dict[synapse_group.target]
                            self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
                                target_group.neuron_channel_assoc_table[neuron_id, 0] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_AMPA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)

                    # external ampa
                    if channel == 'AMPA_EXT':
                        if '[' in synapse_group.target:
                            target_group = neurongroup_dict[synapse_group.target.split('[')[
                                0]]
                            self.code_string += "\t\t#%%  s_AMPA (external) derivatives for synapse group \n"

                            assoc_neuron_labels = synapse_group.target.split('[')[
                                1][:-1].split(':')
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
                                target_group.neuron_channel_assoc_table[neuron_id, 1] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_AMPA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)
                        else:
                            target_group = neurongroup_dict[synapse_group.target]
                            # source = stimuli[synapse_group.source]
                            self.code_string += "\t\t#%%  s_AMPA_EXT derivatives for synapse group \n"
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
                                target_group.neuron_channel_assoc_table[neuron_id, 1] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_AMPA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)

                    # %% GABA
                    if channel == 'GABA':
                        if '[' in synapse_group.target:
                            target_group = neurongroup_dict[synapse_group.target.split('[')[
                                0]]
                            self.code_string += "\t\t#%%  s_GABA (recurrent) derivatives for synapse group \n"

                            assoc_neuron_labels = synapse_group.target.split('[')[
                                1][:-1].split(':')
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
                                target_group.neuron_channel_assoc_table[neuron_id, 3] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_GABA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)
                        else:
                            target_group = neurongroup_dict[synapse_group.target]
                            self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"
                            for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
                                target_group.neuron_channel_assoc_table[neuron_id, 3] = 1
                                self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                                                                                                            neuron_label,
                                                                                                                            channel_state,
                                                                                                                            neuron_label,
                                                                                                                            synapse_group.params[
                                                                                                                                'tau_GABA'],
                                                                                                                            synapse_group.source,
                                                                                                                            neuron_id)

                    # %% NMDA
                    if channel == 'NMDA':
                        if '[' in synapse_group.target:
                            # target_group = neurongroup_dict[synapse_group.target.split('[')[0]]
                            # self.code_string += "\t\t#%%  s_GABA (recurrent) derivatives for synapse group \n"

                            # assoc_neuron_labels = synapse_group.target.split('[')[1][:-1].split(':')
                            # for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
                            #     self.code_string += "\t\tself.d_{}_{} += ((-self.{}_{}) / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                            #                                                                                                     neuron_label,
                            #                                                                                                     channel_state,
                            #                                                                                                     neuron_label,
                            #                                                                                                     synapse_group.params['tau_AMPA'],
                            #                                                                                                     synapse_group.source,
                            #                                                                                                     neuron_id)
                            raise NotImplementedError()
                        else:
                           target_group = neurongroup_dict[synapse_group.target]
                           self.code_string += "\t\t#%%  NMDA (recurrent) derivatives for synapse group \n"
                           for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
                               target_group.neuron_channel_assoc_table[neuron_id, 2] = 1
                               
                               for cs in channel_state.split(','):
                                   if cs.strip() == 'xNMDA':
                                       self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + stimuli[\'{}\'][{}]\n".format(cs.strip(),
                                                                                                                                   neuron_label,
                                                                                                                                   cs.strip(),
                                                                                                                                   neuron_label,
                                                                                                                                   synapse_group.params[
                                           'tau_NMDA_rise'],
                                           synapse_group.source,
                                           neuron_id)
                                   elif cs.strip() == 'sNMDA':
                                       # TODO: yanlış var, xnmda fix
                                       self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + {}*self.{}_{}*(1 - self.{}_{}) \n".format(cs.strip(),
                                                                                                                                               neuron_label,
                                                                                                                                               cs.strip(),
                                                                                                                                               neuron_label,
                                                                                                                                               synapse_group.params[
                                           'tau_NMDA_decay'],
                                           synapse_group.params[
                                           'alpha'],
                                           'xNMDA',
                                           neuron_label,
                                           cs.strip(),
                                           neuron_label
                                       )
                                   else:
                                       raise NotImplementedError()
        
                    
        
            #%% generate current func
            self.code_string += "\t#%% synaptic current calculation func\n"
            self.code_string += "\tdef calculate_synaptic_currents(self):\n"
            
            
            
            # Exc to exc:  cols from, rows to
            #    |   A       B       N
            #  ---------------------------------
            #  A |   w+      w-      w-
            #  B |   w-      w+      w-
            #  N |   1       1       1
            
            for synapse_group in synapses:
                for channel_id, channel in enumerate(synapse_group.channels):
                    channel_state = synapse_group.state_labels[channel_id]
                    # recurrent ampa
                    if channel == 'AMPA':
                        # for every target neurongroup neuron
                        for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
                            self.code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
                            self.code_string += "{}*(self.V_{} - {})*(".format(synapse_group.g_AMPA, target_neuron, neurongroup_dict[synapse_group.target].params['VE'])
                            
                            for wid, w in enumerate(synapse_group.w_AMPA[target_neuron_id]):
                                if wid == 0:
                                    self.code_string += "{}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
                                else:
                                    if w != 0:
                                        self.code_string += " + {}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
                            self.code_string += ")\n"

                    if channel == 'NMDA':
                        # for every target neurongroup neuron
                        for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
                            self.code_string += "\t\tI_{}_{} = ".format(target_neuron, 'sNMDA')
                            self.code_string += "({}*(self.V_{} - {})/(1 + ({}*np.exp(-0.062*self.V_{})/3.57)))*(".format(synapse_group.g_NMDA, 
                                                                                        target_neuron, 
                                                                                        neurongroup_dict[synapse_group.target].params['VE'],
                                                                                        synapse_group.params['C_Mg'],
                                                                                        target_neuron)
                            
                            for wid, w in enumerate(synapse_group.w_NMDA[target_neuron_id]):
                                if wid == 0:
                                    self.code_string += "{}*self.{}_{}".format(w, 'sNMDA', neurongroup_dict[synapse_group.source].neuron_labels[wid])
                                else:
                                    if w != 0:
                                        self.code_string += " + {}*self.{}_{}".format(w, 'sNMDA', neurongroup_dict[synapse_group.source].neuron_labels[wid])
                            self.code_string += ")\n"
                        
                    if channel == 'GABA':
                        # for every target neurongroup neuron
                        for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
                            self.code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
                            self.code_string += "{}*(self.V_{} - {})*(".format(synapse_group.g_GABA, target_neuron, neurongroup_dict[synapse_group.target].params['VE'])
                            
                            for wid, w in enumerate(synapse_group.w_GABA[target_neuron_id]):
                                if wid == 0:
                                    self.code_string += "{}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
                                else:
                                    if w != 0:
                                        self.code_string += " + {}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
                            self.code_string += ")\n"
                        
                    if channel == 'AMPA_EXT':
                        # for every target neurongroup neuron
                        tg = synapse_group.target.split('[')
                        for target_neuron_id, target_neuron in enumerate(neurongroup_dict[tg[0]].neuron_labels):
                            code_string = """"""
                            code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
                            code_string += "{}*(self.V_{} - {})*self.{}_{}\n".format(synapse_group.g_AMPA_ext, target_neuron, neurongroup_dict[tg[0]].params['VE'],
                                                                                          channel_state, target_neuron)
                            
                            if code_string not in self.code_string:
                                self.code_string += code_string
            
            # calc syn currents
            self.code_string += "\t\t#%% calculate synaptic currents\n"
            
            for neuron_group in neurons:
                for neuron_id, neuron_label in enumerate(neuron_group.neuron_labels):
                    assocs = neuron_group.neuron_channel_assoc_table[neuron_id]
                    #assocs ampa ampaext nmda gaba
                    self.code_string += "\t\tself.Isyn_{} = ".format(neuron_label)
                    if assocs[0]:
                        self.code_string += "+ I_{}_{}".format(neuron_label, 'sAMPA')
                    if assocs[1]:
                        self.code_string += "+ I_{}_{}".format(neuron_label, 'sAMPA_EXT')
                    if assocs[2]:
                        self.code_string += "+ I_{}_{}".format(neuron_label, 'sNMDA')
                    if assocs[3]:
                        self.code_string += "+ I_{}_{}".format(neuron_label, 'sGABA')
                    
                    self.code_string += "\n"
            
            #%% generate step func
            self.code_string += "\t#%% step func\n"
            self.code_string += "\tdef step(self):\n"
            
            self.code_string += "\t\tself.calculate_dxdt_all()\n"
            self.code_string += "\t\tself.calculate_synaptic_currents()\n"
            self.code_string += "\t\tself.integrate_all_euler()\n"
            

                                # self.code_string += "\t\tself.d_{}_{} += ((-self.{}_{}) / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
                                #                                                                                                 neuron_label,
                                #                                                                                                 channel_state,
                                #                                                                                                 neuron_label,
                                #                                                                                                 synapse_group.params['tau_AMPA'],
                                #                                                                                                 synapse_group.source,
                                #                                                                                                 neuron_id)

            # for neuron_group_name, neuron_group in neurons.items():
            #     for i in range(neuron_group.no_neurons):
            #         neuron_label = neuron_group_name+str(i)
            #         self.code_string += "\t\t# >>>>>>>>> for neuron {}\n".format(neuron_group_name+str(i))
            #         self.code_string += "\t\tself.dv_{} = (-{}*(self.{}_{} - {}) - I_SYN) / {}\n".format(neuron_label,
            #                                                                                              neuron_group.params['GL'],
            #                                                                                              neuron_group.state_vector[0],
            #                                                                                              neuron_label,
            #                                                                                              neuron_group.params['VL'],
            #                                                                                              neuron_group.params['CM'])

            #         self.code_string += "\t\tds_AMPA_{} = (-self.sAMPA_{}) / {}\n".format(neuron_label, neuron_label, 1.0)  ##TODO: düzelt
                    # dsAmpa = -sAmpa / tau_AMPA
        else:
            print(
                "Simulation stoped! Please first analyse your problem with ProblemGenerator.analyze_network")

        with open("problem.py", "w") as text_file:
            text_file.write(self.code_string)

        return self.code_string


class ODESolver:
    def __init__(self):
        pass


# %% Integrators


class EulerIntegrator:
    def __init__(self):
        pass


class LIFIntegrator(EulerIntegrator):
    def __init__(self):
        super().__init__()

    def lif(u, p, t):
        gL, EL, C, Vth, I = p
        return (-gL*(u-EL)+I)/C

    def threshold(u, t, integrator):
        integrator.u > integrator.p[4]

    def reset(integrator):
        integrator.u = integrator.p[2]


def euler(f, x0, t):
    X = np.zeros((len(t), len(x0)), float)
    dxdt = np.zeros(len(x0), float)

    X[0, :] = x = np.array(x0)
    tlast = t[0]

    for n, tcur in enumerate(t[1:], 1):
        f(x, tlast, dxdt)
        X[n, :] = x = x + dxdt * (tcur - tlast)
        tlast = tcur

    return X
