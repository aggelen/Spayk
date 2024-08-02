#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 21:41:25 2022

@author: aggelen
"""
import numpy as np
from tqdm import tqdm
import networkx as nx
from collections import defaultdict 

from spayk.Configurations import SynapseConfigurator
from spayk.Synapses import COBASynapses
from spayk.Visualization import NetworkVisualizer
from spayk.Utils import ConnectionManager
from spayk.Models import LIFNeuronGroup

# from spayk.Learning import STDP_Engine

from rich import print as rprint
from collections import defaultdict
import pathlib
import pickle

#%% New Core
class CodeGenerator:
    def __init__(self):
        # self.verbose = True
        self.code_string = """"""
        self.has_analyzed = False
        
        self.verbose_analyze = True
        self.verbose_codegen = True

    def analyze_network(self, neurons, synapses, stimului):
        self.runtime_path = "{}/first_run".format(pathlib.Path().resolve())
        pathlib.Path(self.runtime_path).mkdir(parents=True, exist_ok=True)
        
        neuron_group_labels = []
        neuron_group_nos = {}
        total_no_of_neurons = 0
        
        self.neuron_labels = []
        self.neuron_GLs = []
        self.neuron_VLs = []
        self.neuron_CMs = []
        
        self.neuron_groups = {}
        
        stimuli_dict = {}
        for neurongroup in neurons:
            self.neuron_groups[neurongroup.group_label] = neurongroup
            stimuli_dict[neurongroup.group_label] = neurongroup
            neuron_group_labels.append(neurongroup.group_label)
            neuron_group_nos[neurongroup.group_label] = neurongroup.no_neurons
            total_no_of_neurons += neurongroup.no_neurons
            
            for i in range(neurongroup.no_neurons):
                self.neuron_labels.append(neurongroup.neuron_labels[i])
                self.neuron_GLs.append(neurongroup.params['GL'])
                self.neuron_VLs.append(neurongroup.params['VL'])
                self.neuron_CMs.append(neurongroup.params['CM'])
        
        self.neuron_labels = np.array(self.neuron_labels)
        
        self.neuron_GLs = np.array(self.neuron_GLs)
        self.neuron_VLs = np.array(self.neuron_VLs)
        self.neuron_CMs = np.array(self.neuron_CMs)
        
        self.neuron_dict = {'GLs': self.neuron_GLs,
                            'VLs': self.neuron_VLs,
                            'CMs': self.neuron_CMs,
                            'neuron_groups': self.neuron_groups}
        
        
        
        with open('{}/neuron_dict.pickle'.format(self.runtime_path), 'wb') as handle:
            pickle.dump(self.neuron_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        for stim in stimului:
            stimuli_dict[stim.group_label] = stim
            
        if self.verbose_analyze:
            rprint("\n[cyan]-- Spayk is analyzing your configuration. --[/cyan]")
            
            rprint("\n[red]-- Neuron Info[/red] --")
            print("Total No of Neurons: {}".format(total_no_of_neurons))
            for i, ngl in enumerate(neuron_group_labels):
                print("{} neuron in group {}".format(neuron_group_nos[ngl], ngl))
                
                
        if self.verbose_analyze:
            rprint("\n[red]-- Synaptic Configuration Info[/red] --")
            print("Total No of Synaptic Configurations: {}".format(len(synapses)))
            
        connection_list = []
        
        for synapseconfig in synapses:
            if self.verbose_analyze:
                for ch in synapseconfig.channels:
                    
                    if ch == 'AMPA':
                        g = synapseconfig.g_AMPA
                        W = synapseconfig.w_AMPA
                    elif ch == 'NMDA':
                        g = synapseconfig.g_NMDA
                        W = synapseconfig.w_NMDA
                    elif ch == 'GABA':
                        g = synapseconfig.g_GABA
                        W = synapseconfig.w_GABA
                    else:
                        g = synapseconfig.g_AMPA_ext
                        W = synapseconfig.w_AMPA_ext
                    
                    if len(synapseconfig.target.split('[')) > 1:
                        
                        conn = {'from': synapseconfig.source,
                                'to': synapseconfig.target.split('[')[0].strip(),
                                'channel': ch,
                                'g': g,
                                'W': W,
                                'synapse_params': synapseconfig.params,
                                'subgroup_operation': True,
                                'target_subgroup': '['+synapseconfig.target.split('[')[1]}
                        connection_list.append(conn)
                          
                        print("Group {} -> Group {} with {}".format(synapseconfig.source, synapseconfig.target, ch))
                    else:
                        conn = {'from': synapseconfig.source,
                                'to': synapseconfig.target,
                                'channel': ch,
                                'g': g,
                                'W': W,
                                'synapse_params': synapseconfig.params,
                                'subgroup_operation': False,
                                'target_subgroup': None}
                        connection_list.append(conn)
                          
                        print("Group {} -> Group {} with {}".format(synapseconfig.source, synapseconfig.target, ch))
        
        self.connection_list = connection_list
        self.connection_dict = defaultdict(list)
        self.stimuli_dict = stimuli_dict
        
        self.neuron_group_labels = neuron_group_labels
        self.neuron_group_nos = neuron_group_nos
        self.total_no_of_neurons = total_no_of_neurons
        
        for conn in connection_list:
            source_stim = stimuli_dict[conn['from']]
            target_stim = stimuli_dict[conn['to']]
            
            if conn['subgroup_operation']:
                fr = int(conn['target_subgroup'][1:].split(':')[0].strip())
                to = int(conn['target_subgroup'][1:].split(':')[1][:-1].strip())
                for target_label in target_stim.neuron_labels[fr:to]:
                    for source_label in source_stim.neuron_labels:
                        self.connection_dict[target_label].append(conn['channel']+'<'+source_label+"@{}".format(source_stim.group_label))
            else:
                for target_label in target_stim.neuron_labels:
                    for source_label in source_stim.neuron_labels:
                        self.connection_dict[target_label].append(conn['channel']+'<'+source_label+"@{}".format(source_stim.group_label))
                
        
        if self.verbose_analyze:
            rprint("\n[green]-- Network Analyze: OK[/green] --\n")
            
        self.has_analyzed = True
        
        
        
        self.synapse_dict = {'connection_list': connection_list,
                             'connection_dict': self.connection_dict}
        
        with open('{}/synapse_dict.pickle'.format(self.runtime_path), 'wb') as handle:
            pickle.dump(self.synapse_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open('{}/stimuli_dict.pickle'.format(self.runtime_path), 'wb') as handle:
            pickle.dump(self.stimuli_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def count_channels_for_a_neuron(conn_list, channel_str):
        no_of_conns = 0
        conn_index = []
        for conn in conn_list:
            if channel_str in conn:
                no_of_conns += 1
                conn_index.append(conn.split('@')[-1])
        return no_of_conns, conn_index
                
    def make_sysofodes(self, neurons, synapses, stimuli):
        if self.has_analyzed:
            if self.verbose_analyze:
                rprint("\n[cyan]-- Spayk runtime code-generation has started! --[/cyan]")
                
            
            
            #%% create stimuli
            self.code_string += """import numpy as np\n"""
            self.code_string += """from spayk.Stimuli import *\nimport pickle\n"""
            self.code_string += """def load_pickle(pkl_path):\n"""
            self.code_string += """\twith open(pkl_path, 'rb') as handle:\n"""
            self.code_string += """\t\treturn pickle.load(handle)"""
            
            self.code_string += "#%% Problem\n"
            self.code_string += """class Problem:\n\tdef __init__(self):\n"""
            self.code_string += "\t\tself.runtime_path = '{}'\n".format(self.runtime_path)
            self.code_string += "\t\tself.neuron_dict = load_pickle('{}/neuron_dict.pickle')\n".format(self.runtime_path)
            self.code_string += "\t\tself.synapse_dict = load_pickle('{}/synapse_dict.pickle')\n\n".format(self.runtime_path)
            self.code_string += "\t\tself.stimuli_dict = load_pickle('{}/stimuli_dict.pickle')\n\n".format(self.runtime_path)
            self.code_string += "\t\tself.dt = 0.1e-3\n\n"
            self.code_string += "\t\tself.spikes = []\n\n"
            
            self.code_string += """\t\t#%% Stimuli \n"""
            self.code_string += "\t\tself.stimuli = {"
            for stimulus in stimuli:
                if stimulus.activity_type == 'poisson':
                    self.code_string += """'{}': PoissonSpikeTrain({},{},{}),\n""".format(stimulus.group_label, 
                                                                                          stimulus.no_neurons,
                                                                                          stimulus.firing_rates,
                                                                                          stimulus.time_params)
    
                    
            self.code_string += "\t\t}\n"
            
            self.code_string += "\t\tself.current_stimuli = {"
            for stimulus in stimuli:
                if stimulus.activity_type == 'poisson':
                    self.code_string += """'{}': self.stimuli['{}'].spikes[0],\n""".format(stimulus.group_label, stimulus.group_label)
            for neuron_group in neurons: 
                spikes = []
                for neuron_label in neuron_group.neuron_labels:            
                    spikes.append(0)
                self.code_string += """'{}': np.zeros({}),\n""".format(neuron_group.group_label, 
                                                                       neuron_group.no_neurons)
            self.code_string += "\t\t}\n"
            
            #%% generate variables
            self.code_string += "\t\t#%% neuron memb. potens.\n"
            
            # lif variables vectorized
            self.code_string += "\t\t# >>>>>>>>> lif variables\n"
            self.code_string += "\t\tself.V = np.zeros({})\n".format(self.total_no_of_neurons)
            self.code_string += "\t\tself.t_ref = np.zeros({})\n".format(self.total_no_of_neurons)
            self.code_string += "\t\tself.I_syn = np.zeros({})\n".format(self.total_no_of_neurons)
            
            self.code_string += "\t\tself.neuron_GLs = self.neuron_dict['GLs']\n"
            self.code_string += "\t\tself.neuron_VLs = self.neuron_dict['VLs']\n"
            self.code_string += "\t\tself.neuron_CMs = self.neuron_dict['CMs']\n\n"
            
            self.code_string += "\t\t#%% derivatives \n"                       
            for conn_id, conn in enumerate(self.synapse_dict['connection_list']):
                no_neurons_target = self.stimuli_dict[conn['to']].no_neurons
                no_neurons_source = self.stimuli_dict[conn['from']].no_neurons
                if conn['channel'] == 'AMPA':
                    self.code_string += "\t\tself.s_AMPA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'NMDA':
                    self.code_string += "\t\tself.s_NMDA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                    self.code_string += "\t\tself.x_NMDA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'GABA':
                    self.code_string += "\t\tself.s_GABA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'AMPA_EXT':
                    if conn['subgroup_operation']:
                        st = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[0])
                        end = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[1])
                        no_neurons = end-st+1
                        self.code_string += "\t\tself.s_AMPA_EXT__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons)
                    else:
                        self.code_string += "\t\tself.s_AMPA_EXT__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)

            self.code_string += "\n\t\t# derivative reset func\n"  
            self.code_string += "\t\tself.dx_reset()\n"
                    

            #%% generate derivative reset func
            self.code_string += "\t#%% derivative reset func\n"
            self.code_string += "\tdef dx_reset(self):\n"
            
            self.code_string += "\t\tself.d_V = np.zeros({})\n".format(self.total_no_of_neurons)
            
            for conn_id, conn in enumerate(self.synapse_dict['connection_list']):
                no_neurons_target = self.stimuli_dict[conn['to']].no_neurons
                no_neurons_source = self.stimuli_dict[conn['from']].no_neurons
                if conn['channel'] == 'AMPA':
                    self.code_string += "\t\tself.d_s_AMPA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'NMDA':
                    self.code_string += "\t\tself.d_s_NMDA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                    self.code_string += "\t\tself.d_x_NMDA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'GABA':
                    self.code_string += "\t\tself.d_s_GABA__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'AMPA_EXT':
                    if conn['subgroup_operation']:
                        st = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[0])
                        end = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[1])
                        no_neurons = end-st+1
                        self.code_string += "\t\tself.d_s_AMPA_EXT__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons)
                    else:
                        self.code_string += "\t\tself.d_s_AMPA_EXT__CONN_{} = np.zeros(({},{}))\n".format(conn_id, no_neurons_source, no_neurons_target)
                        
            #%% generate euler integration func
            self.code_string += "\t#%% euler integration func\n"
            self.code_string += "\tdef integrate_all_euler(self):\n"
            
            self.code_string += "\t\tself.V += self.d_V*self.dt\n".format(self.total_no_of_neurons)
            
            for conn_id, conn in enumerate(self.synapse_dict['connection_list']):
                no_neurons_target = self.stimuli_dict[conn['to']].no_neurons
                no_neurons_source = self.stimuli_dict[conn['from']].no_neurons
                if conn['channel'] == 'AMPA':
                    self.code_string += "\t\tself.s_AMPA__CONN_{} += self.d_s_AMPA__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'NMDA':
                    self.code_string += "\t\tself.s_NMDA__CONN_{} += self.d_s_NMDA__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons_target)
                    self.code_string += "\t\tself.x_NMDA__CONN_{} += self.d_x_NMDA__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'GABA':
                    self.code_string += "\t\tself.s_GABA__CONN_{} += self.d_s_GABA__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons_target)
                if conn['channel'] == 'AMPA_EXT':
                    if conn['subgroup_operation']:
                        st = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[0])
                        end = int(conn['target_subgroup'].replace('[','').replace(']','').split(':')[1])
                        no_neurons = end-st+1
                        self.code_string += "\t\tself.s_AMPA_EXT__CONN_{} += self.d_s_AMPA_EXT__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons)
                    else:
                        self.code_string += "\t\tself.s_AMPA_EXT__CONN_{} += self.d_s_AMPA_EXT__CONN_{}*self.dt\n".format(conn_id, no_neurons_source, no_neurons_target)
            
            #%% generate derivative func
            self.code_string += "\t#%% derivative func\n"
            self.code_string += "\tdef calculate_dxdt_all(self):\n"

            self.code_string += "\t\tself.dx_reset()\n"
            
            self.code_string += "\t\t##% memb pot derivatives\n"
            
            self.code_string += "\t\tself.d_V = (-self.neuron_GLs*(self.V - self.neuron_VLs) - self.I_syn) / self.neuron_CMs\n"
            
            self.code_string += "\t\t##% channel derivatives\n"
            
            for conn_id, conn in enumerate(self.synapse_dict['connection_list']):
                if conn['channel'] == 'AMPA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---AMPA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---AMPA---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.d_s_AMPA__CONN_{} += (-self.s_AMPA__CONN_{} / self.synapse_dict['connection_list'][0]['synapse_params']['tau_AMPA']) + self.current_stimuli['{}'] \n".format(conn_id, 
                                                                                                                                                conn_id,
                                                                                                                                                conn['from'])
                        
                            
                if conn['channel'] == 'NMDA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---NMDA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---NMDA---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.d_x_NMDA__CONN_{} += (-self.x_NMDA__CONN_{} / self.synapse_dict['connection_list'][0]['synapse_params']['tau_NMDA_rise']) + self.current_stimuli['{}'] \n".format(conn_id, 
                                                                                                                                               conn_id,
                                                                                                                                               conn['from'])
                        self.code_string += "\t\tself.d_s_NMDA__CONN_{} += (-self.s_NMDA__CONN_{} / {}) + {}*self.x_NMDA__CONN_{}*(1 - self.s_NMDA__CONN_{}) \n".format(conn_id, 
                                                                                                                                               conn_id,
                                                                                                                                               self.synapse_dict['connection_list'][conn_id]['synapse_params']['tau_NMDA_decay'],
                                                                                                                                               self.synapse_dict['connection_list'][conn_id]['synapse_params']['alpha'],
                                                                                                                                               conn_id,
                                                                                                                                               conn_id)                        
                       
                     
                if conn['channel'] == 'GABA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---GABA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---GABA---> {}\n".format(conn['from'], conn['to'])
         
                        self.code_string += "\t\tself.d_s_GABA__CONN_{} += (-self.s_GABA__CONN_{} / self.synapse_dict['connection_list'][0]['synapse_params']['tau_GABA']) + self.current_stimuli['{}'] \n".format(conn_id, 
                                                                                                                                                    conn_id,
                                                                                                                                                    conn['from'])
                        
                if conn['channel'] == 'AMPA_EXT':
                    if conn['subgroup_operation']:
                        self.code_string += "\t\t# {} ---AMPA_EXT---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.d_s_AMPA_EXT__CONN_{} += (-self.s_AMPA_EXT__CONN_{} / self.synapse_dict['connection_list'][0]['synapse_params']['tau_AMPA']) + self.current_stimuli['{}'] \n".format(conn_id, 
                                                                                                                                                 conn_id,
                                                                                                                                                 conn['from'])
                    else:
                        self.code_string += "\t\t# {} ---AMPA_EXT---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.d_s_AMPA_EXT__CONN_{} += (-self.s_AMPA_EXT__CONN_{} / self.synapse_dict['connection_list'][0]['synapse_params']['tau_AMPA']) + self.current_stimuli['{}'] \n".format(conn_id, 
                                                                                                                                                 conn_id,
                                                                                                                                              conn['from'])
                
                # if conn['channel'] == 'AMPA_EXT':
                #     if conn['subgroup_operation']:
                #         self.code_string += "\t\t# {} ---AMPA_EXT---subgroup of---> {}\n".format(conn['from'], conn['to'])
                #         for neuron in self.neuron_dict['neuron_groups'][conn['to']].neuron_labels:
                #             connection_index = self.ampa_ext_conn_index_dict[neuron]
                #             conn_filter = np.array(connection_index) == conn['from']
                #             if conn_filter.sum():
                #                 idx_min = min(np.where(conn_filter)[0])
                #                 idx_max = max(np.where(conn_filter)[0])+1
                                
                #                 self.code_string += "\t\tself.d_s_AMPA_EXT_{}[{}:{}] += (-self.s_AMPA_EXT_{}[{}:{}] / {}) + self.current_stimuli['{}'][{}:{}]  \n".format(neuron, 
                #                                                                                           idx_min,
                #                                                                                           idx_max,
                #                                                                                           neuron, 
                #                                                                                           idx_min,
                #                                                                                           idx_max,
                #                                                                                           conn['synapse_params']['tau_AMPA'],
                #                                                                                           conn['from'],
                #                                                                                           idx_min,
                #                                                                                           idx_max)
                #     else:
                #         self.code_string += "\t\t# {} ---AMPA_EXT---> {}\n".format(conn['from'], conn['to'])
                #         for neuron in self.neuron_dict['neuron_groups'][conn['to']].neuron_labels:
                #             connection_index = self.ampa_ext_conn_index_dict[neuron]
                #             conn_filter = np.array(connection_index) == conn['from']
                #             idx_min = min(np.where(conn_filter)[0])
                #             idx_max = max(np.where(conn_filter)[0])+1
                            
                #             self.code_string += "\t\tself.d_s_AMPA_EXT_{}[{}:{}] += (-self.s_AMPA_EXT_{}[{}:{}] / {}) + self.current_stimuli['{}'][{}:{}]  \n".format(neuron, 
                #                                                                                       idx_min,
                #                                                                                       idx_max,
                #                                                                                       neuron, 
                #                                                                                       idx_min,
                #                                                                                       idx_max,
                #                                                                                       conn['synapse_params']['tau_AMPA'],
                #                                                                                       conn['from'],
                #                                                                                       idx_min,
                #                                                                                       idx_max)
                        
            #%% generate current func
            self.code_string += "\t#%% synaptic current calculation func\n"
            self.code_string += "\tdef calculate_synaptic_currents(self):\n"
            
            # Exc to exc:  cols from, rows to
            #    |   A       B       N
            #  ---------------------------------
            #  A |   w+      w-      w-
            #  B |   w-      w+      w-
            #  N |   1       1       1
            
            for conn_id, conn in enumerate(self.connection_list):
                if conn['channel'] == 'AMPA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---AMPA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---AMPA---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.I_Syn += {}*(self.V-VE)*(self.synapse_dict['connection_list'][{}]['W']) \n".format(conn['g'],
                                                                                                                                         conn_id)
                if conn['channel'] == 'NMDA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---AMPA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---NMDA---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.I_Syn += {}*(self.V-VE)*(self.synapse_dict['connection_list'][{}]['W']) \n".format(conn['g'],
                                                                                                                                         conn_id)
                if conn['channel'] == 'GABA':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---AMPA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        raise NotImplementedError()
                    else:
                        self.code_string += "\t\t# {} ---GABA---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.I_Syn += {}*(self.V-VE)*(self.synapse_dict['connection_list'][{}]['W']) \n".format(conn['g'],
                                                                                                                                         conn_id)
                        
                if conn['channel'] == 'AMPA_EXT':
                    if conn['subgroup_operation']:
                        # self.code_string += "\t\t# {} ---AMPA---subgroup of---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\t# {} ---AMPA_EXT---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.I_Syn += {}*(self.V-VE)*(self.synapse_dict['connection_list'][{}]['W']) \n".format(conn['g'],
                                                                                                                                         conn_id)
                    else:
                        self.code_string += "\t\t# {} ---AMPA_EXT---> {}\n".format(conn['from'], conn['to'])
                        self.code_string += "\t\tself.I_Syn += {}*(self.V-VE)*(self.synapse_dict['connection_list'][{}]['W']) \n".format(conn['g'],
                                                                                                                                          conn_id)
                

            
            #%% firing controls
            self.code_string += "\t#%% firing control\n"
            self.code_string += "\tdef fire(self):\n"
            
            for neuron_group in neurons: 
                self.code_string += "\t\tcurrent_spikes_{} = []\n".format(neuron_group.group_label)
            # check resting time
            
            # if tr >0:
            # v[it] = V_reset
            #       tr = tr-1
            # elif v[it] >= V_th:         #reset voltage and record spike event
            #       rec_spikes.append(it)
            #       v[it] = V_reset
            #       tr = tref/dt
            # #calculate the increment of the membrane potential
            # dv = (-(v[it]-V_L) + I[it]/g_L) * (dt/tau_m)
              
            # #update the membrane potential
            # v[it+1] = v[it] + dv
            concat_list = []
            for neuron_group in neurons: 
                for neuron_label in neuron_group.neuron_labels:
                    self.code_string += "\t\tif self.t_ref_{} > 0.0:\n".format(neuron_label)
                    self.code_string += "\t\t\tself.t_ref_{} -= self.dt\n".format(neuron_label)
                    self.code_string += "\t\t\tcurrent_spikes_{}.append(0)\n".format(neuron_group.group_label, neuron_group.group_label)
                    self.code_string += "\t\telif self.{}_{} >= {}:\n".format(neuron_group.state_vector[0], neuron_label, neuron_group.params['VT'])
                    self.code_string += "\t\t\tself.{}_{} = {}\n".format(neuron_group.state_vector[0], neuron_label, neuron_group.params['VR'])
                    self.code_string += "\t\t\tself.t_ref_{} = {}\n".format(neuron_label, neuron_group.params['TREF'])
                    self.code_string += "\t\t\tcurrent_spikes_{}.append(1)\n".format(neuron_group.group_label, neuron_group.group_label)
                self.code_string += "\t\tself.current_stimuli['{}'] = current_spikes_{}\n".format(neuron_group.group_label, neuron_group.group_label)
                concat_list.append('current_spikes_{}'.format(neuron_group.group_label))
                
            self.code_string += """\t\tself.spikes.append("""
            for cli, cl in enumerate(concat_list):
                if cli == 0:
                    self.code_string += """{} """.format(cl)
                else:
                    self.code_string += """+ {}""".format(cl)
            self.code_string += """)\n"""
            
            #%% data acquisition
            # data includes spikes for only the current time step with same labels
            self.code_string += "\t#%% data acquisition\n"
            self.code_string += "\tdef data_acquisition(self, time_step):\n"
            
            self.code_string += "\t\tself.data = {}\n"
            

            #%% generate step func
            self.code_string += "\t#%% step func\n"
            self.code_string += "\tdef step(self):\n"
            
            self.code_string += "\t\tself.calculate_dxdt_all()\n"
            self.code_string += "\t\tself.calculate_synaptic_currents()\n"
            self.code_string += "\t\tself.integrate_all_euler()\n"
            
            #%% run the network
            self.code_string += "#%% Solution\n"
            self.code_string += "problem = Problem()\n"
            self.code_string += "problem.calculate_dxdt_all()\n"
        
        else:
            print(
                "Simulation stoped! Please first analyse your problem with ProblemGenerator.analyze_network")

        with open("problem.py", "w") as text_file:
            text_file.write(self.code_string)

        return self.code_string




#%% Old Core
class Simulator:
    def __init__(self):
        """
        Simulator module!
        """
        # self.a = 0.02
        # self.b = 0.25
        # self.c = -65
        # self.d = 6
        # self.vt = 30    #mv
        pass
    
    def new_core(self, tissue, params):
        
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        steps = range(int(T / dt))
        for step in steps:
            t = step*dt
            
            # We generate a current step of 7 mA between 200 and 700 ms
            if t > 200 and t < 700:
                i_in = 7.0
            else:
                i_in = 0.0
            
            tissue.inject_current(i_in)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(i_in)
            self.results['v_out'].append(v)
            
    def new_core_syn(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'presyn_spikes': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            if np.sum(p_syn_spike) > 0:
                aykut = 1
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            self.results['presyn_spikes'].append(p_syn_spike)
            
    def new_core_syn_experimental(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'presyn_spikes': [],
                        'weight_means': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            self.results['presyn_spikes'].append(p_syn_spike)
            # self.results['weight_means'].append([tissue.W.mean(), tissue.W_in.mean()])
            self.results['weight_means'].append(tissue.W_in.mean())
            
    def new_core_syn_stdp(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        
        w_prev = tissue.W_in
        delta_weights = np.zeros((int(T / dt), tissue.no_connected_neurons))
    
        
        for step in steps:
            t = step*dt
            
            if 'stimuli' in params.keys():
                p_syn_spike = params['stimuli'][step]
            else:
                if t > 200 and t < 700:
                    # Generate a random matrix
                    r = np.random.uniform(0,1,(tissue.no_connected_neurons))
                    # A synapse has spiked when r is lower than the spiking rate
                    p_syn_spike = r < params['frate'] * params['dt']
                else:
                    # No synapse activity during that period
                    p_syn_spike = np.zeros((tissue.no_connected_neurons), dtype=bool)
            
            tissue.inject_spike_train(p_syn_spike)
            v, u = tissue()
            ConnectionManager
            # Store values
            self.results['I_in'].append(tissue.I)
            self.results['v_out'].append(v)
            
            w_next = tissue.W_in
            delta_weights[step,:] = w_next - w_prev
            w_prev = w_next
            
        self.results['delta_weights'] = delta_weights
        
    def keep_alive(self, organizations, settings):
        dt = settings['dt']
        time = np.arange(0,settings['duration'],dt)
        
        if 'synaptic_plasticity' in settings.keys():
            self.synaptic_plasticity = settings['synaptic_plasticity']
        else:
            raise Exception('synaptic_plasticity status must be set!')
        
        #FIXME! Worst soln ever!
        # for neuron in organization.neurons:
        #     for t in time:
        #         neuron.forward(neuron.stimuli.I, dt)
                
        for t_id in tqdm(range(time.shape[0])):
            t = time[t_id]
            for organization in organizations:
                self.izhikevich_update(organization, t, dt)
                
        for organization in organizations:
            organization.end_of_life()
                
    def izhikevich_update(self, organization, t, dt):
        vs, us, dMat = organization.vs, organization.us, organization.dynamics_matrix
        
        Is = organization.calculate_Is(t, self.synaptic_plasticity)
        
        a,b,c,d,vt = dMat[:,0],dMat[:,1],dMat[:,2],dMat[:,3],dMat[:,4]
        
        dv = 0.04*np.square(vs) + 5*vs + 140 - us + Is
        du = a*(b*vs - us)
        vs = vs + dv*dt
        us = us + du*dt

        spikes = np.greater_equal(vs,vt)
        
        #FIXME: delete org name
        # if any(spikes):
        if organization.stdp_status and organization.name=='network' and t > 0.0:
            organization.LTD_update(spikes, dt)
    
        us = np.where(spikes, us + d, us)
        vs = np.where(spikes, c, vs)
        
        organization.vs = vs
        organization.us = us
        organization.keep_log(spikes, Is)
        
    def integrate_and_fire(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': []}
        
        dt = params['dt']
        T = params['t_stop']
        
        steps = range(int(T / dt))
        for step in tqdm(steps):
            p_syn_spikes = params['stimuli'][step]
            
            v = tissue(p_syn_spikes)
            self.results['v_out'].append(v)
            
    def integrate_and_fire_stdp(self, tissue, params):
        self.results = {'I_in': [],
                        'v_out': [],
                        'delta_w': [],
                        'mean_w': []}
        
        dt = params['dt']
        T = params['t_stop']
        steps = range(int(T / dt))
        w_prev = tissue.w
        
        for step in tqdm(steps):
            p_syn_spikes = params['stimuli'][step]
            
            v = tissue(p_syn_spikes)
            self.results['v_out'].append(v)
            self.results['mean_w'].append(tissue.w.mean())
            
            w_next = tissue.w
            delta_weights = w_next - w_prev
            w_prev = w_next
            self.results['delta_w'].append(delta_weights)
            
#%% Forward Time Step Based Simulator
# Experimental step-based simulator. It may run very slowly.

class VectorizedLIFNeuralNetwork:
    def __init__(self, params):
        self.dt = params['dt']
        self.no_neurons = params['no_neurons']
        self.no_stimuli = params['no_stimuli']
        
        # self.neuron_configuration = params['neuron_configuration']
        
        # self.connection_manager = None
        
        self.neuron_group = LIFNeuronGroup(params)
        
        # self.graph = nx.Graph()
        # self.node_color_map = []
        
        # self.stimuli_node_idx = []
        
        # #nodes starts from stimuli
        # for i in range(self.no_stimuli):
        #     self.graph.add_node(i, type='stimuli')
        #     self.node_color_map.append('red')
        #     self.stimuli_node_idx.append()
            
        # for i in range(i+1, i+self.no_neurons+1):
        #     self.graph.add_node(i, type='neuron')
        #     self.node_color_map.append('blue')
            
        # nx.draw(self.graph, node_color=self.node_color_map, with_labels=True)
        
 
    def __no_neurons__(self):
        return self.no_neurons
    
class DiscreteNeuralNetwork:
    def __init__(self, dt):
        self.dt = dt
        self.neuron_id_counter = 0
        
        self.neuron_list = []
        self.external_sources = []
        
        self.visualizer = NetworkVisualizer()
        # self.connection_manager = ConnectionManager(self)
        
    def add_neuron(self, neurons):
        for neuron in neurons:
            self.neuron_list.append(neuron)
            self.neuron_id_counter += 1
            
            self.visualizer.add_node(neuron.visual_style, neuron.visual_position)
            
    def add_externals(self, externals):
        for ext in externals:
            self.external_sources.append(ext)
            
            
    def add_connection(self, connection):
        s = connection.rstrip().split(";")
        current_config = SynapseConfigurator(self.dt, self.neuron_list[int(s[0][1:])].type)
        for cnn in s[1:]:
            ch, target = cnn.split('@')
            
            if ch.strip() == "EXT_AMPA":
                target_stim_id = int(target[1:])
                current_config.create_external_AMPA_channel(self.external_sources[target_stim_id])
                
            if ch.strip() == "REC_AMPA":
                target_neuron_id = int(target[1:])
                current_config.create_recurrent_AMPA_channel(self.neuron_list[target_neuron_id])
                
            if ch.strip() == "REC_GABA":
                target_neuron_id = int(target[1:])
                current_config.create_recurrent_GABA_channel(self.neuron_list[target_neuron_id])
                
            if ch.strip() == "REC_NMDA":
                target_neuron_id = int(target[1:])
                current_config.create_recurrent_NMDA_channel(self.neuron_list[target_neuron_id])
                
        synapses = COBASynapses(current_config.generate_config())
        self.neuron_list[int(s[0][1:])].synapses = synapses
                    
    def __no_neurons__(self):
        return len(self.neuron_list)

#%% Logger
class Logger:
    def __init__(self):
        self.neuron_v_traces = None
        self.neuron_spikes = []
        
    def generate_traces(self, no_neurons, no_time_steps):
        self.neuron_v_traces = np.empty((no_neurons, no_time_steps))
        self.neuron_I_traces = np.empty((no_neurons, no_time_steps))
        
    def update_neuron_traces(self, neuron_id, time_step, v, I):
        self.neuron_v_traces[neuron_id, time_step] = v
        self.neuron_I_traces[neuron_id, time_step] = I          #total synaptic current
    
    def add_spikes(self, spikes):
        self.neuron_spikes.append(spikes)
        
    
    
#%% Simulator

class Simulator:
    def __init__(self):
        pass
    
    def create_time(self, t_stop):
        return np.arange(0, t_stop, self.dt)

class DiscreteTimeSimulator:
    def __init__(self, dt):
        self.dt = dt
        self.neural_network = None
        
        self.simulation_log = Logger()
        
        #FIXME : need better logger
        self.v_logs = []
        self.I_logs = []
        
    def configure_neural_network(self, neural_network):
        self.neural_network = neural_network
        
    def create_time(self, t_stop):
        return np.arange(0, t_stop, self.dt)
        
    def keep_alive(self, t_stop):
        self.time_hist = self.create_time(t_stop)
        self.simulation_log.time_hist = self.time_hist
        
        self.simulation_log.generate_traces(self.neural_network.__no_neurons__(), len(self.time_hist))

        
        if self.neural_network is not None:
            # Main Loop
            total = len(self.time_hist)
            with tqdm(total=total, unit="time-step") as pbar:
                for time_step, t in enumerate(self.time_hist):
                    neuron_v_logs = []
                    neuron_I_logs = []
                    for neuron_id, neuron in enumerate(self.neural_network.neuron_list):
                        #each neuron has a pre-configured channel stack, calculate each channels current
                        
                        last_postsynaptic_spike = neuron.spikes[-1] if len(neuron.spikes) else 0
                        neuron.synapses.spiked = last_postsynaptic_spike
                        
                        I_syn = neuron.calculate_synaptic_current(time_step, t) - 0.9e-9
                        v = neuron(I_syn)
                        
                        self.simulation_log.update_neuron_traces(neuron_id, time_step, v*1e3, I_syn)
                        
                        
                    pbar.update(1)
                    
                for n in self.neural_network.neuron_list:
                    self.simulation_log.add_spikes(n.spikes)

        else:
            print("ERROR: There is no neural network configured!")
            
#%% Vectorized
class VectorizedSimulator(Simulator):
    def __init__(self, params):
        super().__init__()
        self.dt = params['dt']
        self.neural_network = params['neural_network']
        self.stimuli = params['stimuli']
        
        self.simulation_log = Logger()
        
        self.v_traces = []
        self.spikes = []
        
        self.I_ext_traces = []
        self.I_rec_ampa_traces = []
        self.I_rec_nmda_traces = []
        self.I_rec_gaba_traces = []
        
    def keep_alive(self, t_stop):
        self.time_hist = self.create_time(t_stop)
        
        total = len(self.time_hist)
        with tqdm(total=total, unit="time-step") as pbar:
            for time_step, t in enumerate(self.time_hist):
                
                # self.synapses['stimuli']
                # self.synapses['recurrent']
                
                I_ext_ampa = self.neural_network.synapses.calculate_external_synaptic_currents(self.stimuli[:, time_step],   #presyn spikes
                                                                                               self.neural_network.spiked,   #postsyn spikes
                                                                                               self.neural_network.v,
                                                                                               time_step, 
                                                                                               t)
                
                # rec_currents = self.neural_network.synapses.calculate_recurrent_synaptic_currents(self.stimuli[:, time_step],   #presyn spikes
                #                                                                                   self.neural_network.spiked,   #postsyn spikes
                #                                                                                   self.neural_network.v,
                #                                                                                   time_step, 
                #                                                                                   t)
                # I_rec_ampa, I_rec_nmda, I_rec_gaba = rec_currents
                
                self.I_ext_traces.append(I_ext_ampa)
                # self.I_rec_ampa_traces.append(I_rec_ampa)
                # self.I_rec_nmda_traces.append(I_rec_nmda)
                # self.I_rec_gaba_traces.append(I_rec_gaba)
                
                # I_syn = np.full(10, -6e-10) + I_ext_ampa + I_rec_ampa + I_rec_nmda + I_rec_gaba
                I_syn = I_ext_ampa 
                # I_syn = I_ext_ampa + I_rec_ampa + I_rec_nmda + I_rec_gaba
                
                v = self.neural_network(I_syn)
                
                self.v_traces.append(v)
                self.spikes.append(self.neural_network.spiked)
                    
                pbar.update(1)
                