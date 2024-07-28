#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 20:40:42 2024

@author: gelenag
"""


# for conn_id, conn in enumerate(self.connection_list):
#     if conn['channel'] == "AMPA":
#         if conn['subgroup_operation']:
#             raise NotImplementedError()
#         else:                        
#             self.code_string += "\t\t# >>>>>>>>> for connection id {}\n".format(conn_id)
#             for neuron_label in self.stimuli_dict[conn['to']].neuron_labels:
#                 self.code_string += "\t\tself.d_s_AMPA_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons)
                
#     elif conn['channel'] == "NMDA":
#         if conn['subgroup_operation']:
#             raise NotImplementedError()
#         else:                        
#             self.code_string += "\t\t# >>>>>>>>> for connection id {}\n".format(conn_id)
#             for neuron_label in self.stimuli_dict[conn['to']].neuron_labels:
#                 self.code_string += "\t\tself.d_s_NMDA_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons)
#                 self.code_string += "\t\tself.d_x_NMDA_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons) 
                
#     elif conn['channel'] == "GABA":
#         if conn['subgroup_operation']:
#             raise NotImplementedError()
#         else:                        
#             self.code_string += "\t\t# >>>>>>>>> for connection id {}\n".format(conn_id)
#             for neuron_label in self.stimuli_dict[conn['to']].neuron_labels:
#                 self.code_string += "\t\tself.d_s_GABA_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons)  
    
#     elif conn['channel'] == "AMPA_EXT":
#         if conn['subgroup_operation']:
#             fr = int(conn['target_subgroup'][1:].split(':')[0].strip())
#             to = int(conn['target_subgroup'][1:].split(':')[1][:-1].strip())
            
#             self.code_string += "\t\t# >>>>>>>>> for connection id {}\n".format(conn_id)
#             for neuron_label in self.stimuli_dict[conn['to']].neuron_labels[fr:to]:
#                 self.code_string += "\t\tself.d_s_AMPA_EXT_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons)  
#         else:                        
#             self.code_string += "\t\t# >>>>>>>>> for connection id {}\n".format(conn_id)
#             for neuron_label in self.stimuli_dict[conn['to']].neuron_labels:
#                 self.code_string += "\t\tself.d_s_AMPA_EXT_{} = np.zeros({})\n".format(neuron_label, self.stimuli_dict[conn['from']].no_neurons)  
                
                
        # for state_label in neuron_group.state_vector:
        #     self.code_string += "\t\tself.{}_{} = 0.0\n".format(
        #         state_label, neuron_label)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        # #%% generate derivative reset func
        # self.code_string += "\t#%% derivative reset func\n"
        # self.code_string += "\tdef dx_reset(self):\n"
        # for neuron_group in neurons:
        #     neurongroup_dict[neuron_group.group_label] = neuron_group
        #     for neuron_label in neuron_group.neuron_labels:
        #         for state_label in neuron_group.state_vector:
        #             self.code_string += "\t\tself.d_{}_{} = 0.0\n".format(
        #                 state_label, neuron_label)
                    
        # #%% generate euler integration func
        # self.code_string += "\t#%% euler integration func\n"
        # self.code_string += "\tdef integrate_all_euler(self):\n"
        # for neuron_group in neurons:
        #     neurongroup_dict[neuron_group.group_label] = neuron_group
        #     for neuron_label in neuron_group.neuron_labels:
        #         for state_label in neuron_group.state_vector:
        #             self.code_string += "\t\tself.{}_{} += self.d_{}_{}*self.dt\n".format(
        #                 state_label, neuron_label,state_label, neuron_label)

        # #%% generate derivative func
        # self.code_string += "\t#%% derivative func\n"
        # self.code_string += "\tdef calculate_dxdt_all(self):\n"

        # self.code_string += "\t\tself.dx_reset()\n"
        
        # self.code_string += "\t\t##% memb pot derivatives\n"
        # for neuron_group in neurons:
        #     for neuron_label in neuron_group.neuron_labels:
        #         self.code_string += "\t\tself.d_{}_{} = (-{}*(self.{}_{} - {}) - self.Isyn_{}) / {}\n".format(neuron_group.state_vector[0], 
        #                                                                                                neuron_label,
        #                                                                                                neuron_group.params['GL'],
        #                                                                                                neuron_group.state_vector[0],
        #                                                                                                neuron_label,
        #                                                                                                neuron_group.params['VL'],
        #                                                                                                neuron_label,
        #                                                                                                neuron_group.params['CM'])
        
        # self.code_string += "\t\t##% channel derivatives\n"
        # for synapse_group in synapses:
        #     for channel_id, channel in enumerate(synapse_group.channels):
        #         channel_state = synapse_group.state_labels[channel_id]
        #         # recurrent ampa
        #         if channel == 'AMPA':
        #             if '[' in synapse_group.target:
        #                 target_group = neurongroup_dict[synapse_group.target.split('[')[0]]
                        
        #                 self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"

        #                 assoc_neuron_labels = synapse_group.target.split('[')[
        #                     1][:-1].split(':')
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 0] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_AMPA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)
        #             else:
        #                 target_group = neurongroup_dict[synapse_group.target]
        #                 self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 0] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_AMPA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)

        #         # external ampa
        #         if channel == 'AMPA_EXT':
        #             if '[' in synapse_group.target:
        #                 target_group = neurongroup_dict[synapse_group.target.split('[')[
        #                     0]]
        #                 self.code_string += "\t\t#%%  s_AMPA (external) derivatives for synapse group \n"

        #                 assoc_neuron_labels = synapse_group.target.split('[')[
        #                     1][:-1].split(':')
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 1] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_AMPA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)
        #             else:
        #                 target_group = neurongroup_dict[synapse_group.target]
        #                 # source = stimuli[synapse_group.source]
        #                 self.code_string += "\t\t#%%  s_AMPA_EXT derivatives for synapse group \n"
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 1] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_AMPA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)

        #         # %% GABA
        #         if channel == 'GABA':
        #             if '[' in synapse_group.target:
        #                 target_group = neurongroup_dict[synapse_group.target.split('[')[
        #                     0]]
        #                 self.code_string += "\t\t#%%  s_GABA (recurrent) derivatives for synapse group \n"

        #                 assoc_neuron_labels = synapse_group.target.split('[')[
        #                     1][:-1].split(':')
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 3] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_GABA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)
        #             else:
        #                 target_group = neurongroup_dict[synapse_group.target]
        #                 self.code_string += "\t\t#%%  s_AMPA (recurrent) derivatives for synapse group \n"
        #                 for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
        #                     target_group.neuron_channel_assoc_table[neuron_id, 3] = 1
        #                     self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 channel_state,
        #                                                                                                                 neuron_label,
        #                                                                                                                 synapse_group.params[
        #                                                                                                                     'tau_GABA'],
        #                                                                                                                 synapse_group.source,
        #                                                                                                                 neuron_id)

        #         # %% NMDA
        #         if channel == 'NMDA':
        #             if '[' in synapse_group.target:
        #                 # target_group = neurongroup_dict[synapse_group.target.split('[')[0]]
        #                 # self.code_string += "\t\t#%%  s_GABA (recurrent) derivatives for synapse group \n"

        #                 # assoc_neuron_labels = synapse_group.target.split('[')[1][:-1].split(':')
        #                 # for neuron_id, neuron_label in enumerate(target_group.neuron_labels[int(assoc_neuron_labels[0]):int(assoc_neuron_labels[1])]):
        #                 #     self.code_string += "\t\tself.d_{}_{} += ((-self.{}_{}) / {}) + stimuli[\'{}\'][{}]\n".format(channel_state,
        #                 #                                                                                                     neuron_label,
        #                 #                                                                                                     channel_state,
        #                 #                                                                                                     neuron_label,
        #                 #                                                                                                     synapse_group.params['tau_AMPA'],
        #                 #                                                                                                     synapse_group.source,
        #                 #                                                                                                     neuron_id)
        #                 raise NotImplementedError()
        #             else:
        #                target_group = neurongroup_dict[synapse_group.target]
        #                self.code_string += "\t\t#%%  NMDA (recurrent) derivatives for synapse group \n"
        #                for neuron_id, neuron_label in enumerate(target_group.neuron_labels):
        #                    target_group.neuron_channel_assoc_table[neuron_id, 2] = 1
                           
        #                    for cs in channel_state.split(','):
        #                        if cs.strip() == 'xNMDA':
        #                            self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + self.current_stimuli[\'{}\'][{}]\n".format(cs.strip(),
        #                                                                                                                        neuron_label,
        #                                                                                                                        cs.strip(),
        #                                                                                                                        neuron_label,
        #                                                                                                                        synapse_group.params[
        #                                'tau_NMDA_rise'],
        #                                synapse_group.source,
        #                                neuron_id)
        #                        elif cs.strip() == 'sNMDA':
        #                            # TODO: yanlış var, xnmda fix
        #                            self.code_string += "\t\tself.d_{}_{} += (-self.{}_{} / {}) + {}*self.{}_{}*(1 - self.{}_{}) \n".format(cs.strip(),
        #                                                                                                                                    neuron_label,
        #                                                                                                                                    cs.strip(),
        #                                                                                                                                    neuron_label,
        #                                                                                                                                    synapse_group.params[
        #                                'tau_NMDA_decay'],
        #                                synapse_group.params[
        #                                'alpha'],
        #                                'xNMDA',
        #                                neuron_label,
        #                                cs.strip(),
        #                                neuron_label
        #                            )
        #                        else:
        #                            raise NotImplementedError()
    
                
    
        # #%% generate current func
        # self.code_string += "\t#%% synaptic current calculation func\n"
        # self.code_string += "\tdef calculate_synaptic_currents(self):\n"
        
        
        
        # # Exc to exc:  cols from, rows to
        # #    |   A       B       N
        # #  ---------------------------------
        # #  A |   w+      w-      w-
        # #  B |   w-      w+      w-
        # #  N |   1       1       1
        
        # for synapse_group in synapses:
        #     for channel_id, channel in enumerate(synapse_group.channels):
        #         channel_state = synapse_group.state_labels[channel_id]
        #         # recurrent ampa
        #         if channel == 'AMPA':
        #             # for every target neurongroup neuron
        #             for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
        #                 self.code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
        #                 self.code_string += "{}*(self.V_{} - {})*(".format(synapse_group.g_AMPA, target_neuron, neurongroup_dict[synapse_group.target].params['VE'])
                        
        #                 for wid, w in enumerate(synapse_group.w_AMPA[target_neuron_id]):
        #                     if wid == 0:
        #                         self.code_string += "{}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                     else:
        #                         if w != 0:
        #                             self.code_string += " + {}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                 self.code_string += ")\n"

        #         if channel == 'NMDA':
        #             # for every target neurongroup neuron
        #             for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
        #                 self.code_string += "\t\tI_{}_{} = ".format(target_neuron, 'sNMDA')
        #                 self.code_string += "({}*(self.V_{} - {})/(1 + ({}*np.exp(-0.062*self.V_{})/3.57)))*(".format(synapse_group.g_NMDA, 
        #                                                                             target_neuron, 
        #                                                                             neurongroup_dict[synapse_group.target].params['VE'],
        #                                                                             synapse_group.params['C_Mg'],
        #                                                                             target_neuron)
                        
        #                 for wid, w in enumerate(synapse_group.w_NMDA[target_neuron_id]):
        #                     if wid == 0:
        #                         self.code_string += "{}*self.{}_{}".format(w, 'sNMDA', neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                     else:
        #                         if w != 0:
        #                             self.code_string += " + {}*self.{}_{}".format(w, 'sNMDA', neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                 self.code_string += ")\n"
                    
        #         if channel == 'GABA':
        #             # for every target neurongroup neuron
        #             for target_neuron_id, target_neuron in enumerate(neurongroup_dict[synapse_group.target].neuron_labels):
        #                 self.code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
        #                 self.code_string += "{}*(self.V_{} - {})*(".format(synapse_group.g_GABA, target_neuron, neurongroup_dict[synapse_group.target].params['VE'])
                        
        #                 for wid, w in enumerate(synapse_group.w_GABA[target_neuron_id]):
        #                     if wid == 0:
        #                         self.code_string += "{}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                     else:
        #                         if w != 0:
        #                             self.code_string += " + {}*self.{}_{}".format(w, channel_state, neurongroup_dict[synapse_group.source].neuron_labels[wid])
        #                 self.code_string += ")\n"
                    
        #         if channel == 'AMPA_EXT':
        #             # for every target neurongroup neuron
        #             tg = synapse_group.target.split('[')
        #             for target_neuron_id, target_neuron in enumerate(neurongroup_dict[tg[0]].neuron_labels):
        #                 code_string = """"""
        #                 code_string += "\t\tI_{}_{} = ".format(target_neuron, channel_state)
        #                 code_string += "{}*(self.V_{} - {})*self.{}_{}\n".format(synapse_group.g_AMPA_ext, target_neuron, neurongroup_dict[tg[0]].params['VE'],
        #                                                                               channel_state, target_neuron)
                        
        #                 if code_string not in self.code_string:
        #                     self.code_string += code_string
        
        # # calc syn currents
        # self.code_string += "\t\t#%% calculate synaptic currents\n"
        
        # for neuron_group in neurons:
        #     for neuron_id, neuron_label in enumerate(neuron_group.neuron_labels):
        #         assocs = neuron_group.neuron_channel_assoc_table[neuron_id]
        #         #assocs ampa ampaext nmda gaba
        #         self.code_string += "\t\tself.Isyn_{} = ".format(neuron_label)
        #         if assocs[0]:
        #             self.code_string += "+ I_{}_{}".format(neuron_label, 'sAMPA')
        #         if assocs[1]:
        #             self.code_string += "+ I_{}_{}".format(neuron_label, 'sAMPA_EXT')
        #         if assocs[2]:
        #             self.code_string += "+ I_{}_{}".format(neuron_label, 'sNMDA')
        #         if assocs[3]:
        #             self.code_string += "+ I_{}_{}".format(neuron_label, 'sGABA')
                
        #         self.code_string += "\n"
        
        # #%% firing controls
        # self.code_string += "\t#%% firing control\n"
        # self.code_string += "\tdef fire(self):\n"
        
        # for neuron_group in neurons: 
        #     self.code_string += "\t\tcurrent_spikes_{} = []\n".format(neuron_group.group_label)
        # # check resting time
        
        # # if tr >0:
        # # v[it] = V_reset
        # #       tr = tr-1
        # # elif v[it] >= V_th:         #reset voltage and record spike event
        # #       rec_spikes.append(it)
        # #       v[it] = V_reset
        # #       tr = tref/dt
        # # #calculate the increment of the membrane potential
        # # dv = (-(v[it]-V_L) + I[it]/g_L) * (dt/tau_m)
          
        # # #update the membrane potential
        # # v[it+1] = v[it] + dv
        # concat_list = []
        # for neuron_group in neurons: 
        #     for neuron_label in neuron_group.neuron_labels:
        #         self.code_string += "\t\tif self.t_ref_{} > 0.0:\n".format(neuron_label)
        #         self.code_string += "\t\t\tself.t_ref_{} -= self.dt\n".format(neuron_label)
        #         self.code_string += "\t\t\tcurrent_spikes_{}.append(0)\n".format(neuron_group.group_label, neuron_group.group_label)
        #         self.code_string += "\t\telif self.{}_{} >= {}:\n".format(neuron_group.state_vector[0], neuron_label, neuron_group.params['VT'])
        #         self.code_string += "\t\t\tself.{}_{} = {}\n".format(neuron_group.state_vector[0], neuron_label, neuron_group.params['VR'])
        #         self.code_string += "\t\t\tself.t_ref_{} = {}\n".format(neuron_label, neuron_group.params['TREF'])
        #         self.code_string += "\t\t\tcurrent_spikes_{}.append(1)\n".format(neuron_group.group_label, neuron_group.group_label)
        #     self.code_string += "\t\tself.current_stimuli['{}'] = current_spikes_{}\n".format(neuron_group.group_label, neuron_group.group_label)
        #     concat_list.append('current_spikes_{}'.format(neuron_group.group_label))
            
        # self.code_string += """\t\tself.spikes.append("""
        # for cli, cl in enumerate(concat_list):
        #     if cli == 0:
        #         self.code_string += """{} """.format(cl)
        #     else:
        #         self.code_string += """+ {}""".format(cl)
        # self.code_string += """)\n"""
        
        # #%% data acquisition
        # # data includes spikes for only the current time step with same labels
        # self.code_string += "\t#%% data acquisition\n"
        # self.code_string += "\tdef data_acquisition(self, time_step):\n"
        
        # self.code_string += "\t\tself.data = {}\n"
        

        # #%% generate step func
        # self.code_string += "\t#%% step func\n"
        # self.code_string += "\tdef step(self):\n"
        
        # self.code_string += "\t\tself.calculate_dxdt_all()\n"
        # self.code_string += "\t\tself.calculate_synaptic_currents()\n"
        # self.code_string += "\t\tself.integrate_all_euler()\n"
        
        # #%% run the network
        # self.code_string += "#%% Solution\n"
        # self.code_string += "problem = Problem()\n"
    
    else:
        print(
            "Simulation stoped! Please first analyse your problem with ProblemGenerator.analyze_network")

    with open("problem.py", "w") as text_file:
        text_file.write(self.code_string)

    return self.code_string
