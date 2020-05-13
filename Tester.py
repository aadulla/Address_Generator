import numpy as np
import copy
import math
import random

from Controller import Controller
from Manager import Controller_Manager

class Tester:
    
    def __init__(self):
        pass

    def prefetch_experiment(self, factor):
        num_tiles = len(self.loop_tiling_lst)

        base_factor = factor**(num_tiles-1)
        for dim_dict in self.loop_tiling_lst[0]:
            for key, val in dim_dict.items():
                dim_dict[key] = int(val/base_factor)

        for tile in self.loop_tiling_lst[1:]:
            for dim_dict in tile:
                for key, val in dim_dict.items():
                    dim_dict[key] = int(val*factor)


    def setup(self, 
              loop_tiling_lst, 
              costs, 
              expansion_factor, 
              write_backs, 
              prefetch_factor=None,
              parallel_for_dims=None,
              debug=False):

        num_channels = 1
        num_filters = 1
        num_weights = 1
        num_outputs = 1
        
        # create the memory sizes for the input, weight, and output memories
        input_memory_sizes = []
        weight_memory_sizes = []
        output_memory_sizes = []
        for loop_tile in loop_tiling_lst[::-1]:
            for dim in loop_tile:
                dim_name, dim_size = list(dim.items())[0]
                if dim_name == "channel":
                    num_channels *= dim_size
                elif dim_name == "filter":
                    num_filters  *= dim_size
                elif dim_name == "weight":
                    num_weights  *= dim_size
                elif dim_name == "output":
                    num_outputs  *= dim_size

                num_inputs = num_weights + num_outputs - 1
            
            input_memory_sizes.insert(0, int(num_channels * num_inputs * (1 + expansion_factor)))
            weight_memory_sizes.insert(0, int(num_channels * num_filters * num_weights * (1 + expansion_factor)))
            output_memory_sizes.insert(0, int(num_filters * num_outputs * (1 + expansion_factor)))
        
        # if in debug mode, initialize the base inputs, weights, and memories to be [0,1,2,...]
        if debug:
            input_data = np.arange(num_channels * num_inputs)
            weight_data = np.arange(num_channels * num_filters * num_weights)
            output_data = np.zeros(num_filters * num_outputs)

            input_data = np.reshape(input_data, (num_channels, num_inputs))
            weight_data = np.reshape(weight_data, (num_channels, num_filters, num_weights))
            output_data = np.reshape(output_data, (num_filters, num_outputs))
        
        # if not in debug mode, initialize the base inputs, weights, and memories to be random   
        else:  
            input_data = np.random.randint(10, size=(num_channels, num_inputs))
            weight_data = np.random.randint(10, size=(num_channels, num_filters, num_weights))
            output_data = np.zeros((num_filters, num_outputs))

        print("Total Input Channels: ", len(input_data))
        print("Num Inputs per Channel: ", len(input_data[0]))
        print("Total Weight Filters: ", len(weight_data[0]))
        print("Num Channels per Filter: ", len(weight_data))
        print("Num Weights per Filter: ", len(weight_data[0][0]))
        print("Total Output Channels: ", len(output_data))
        print("Num Outputs per Channel: ", len(output_data[0]))
        print()
        
        print("INPUT DATA:")
        print(input_data)
        print()
        print("INPUT MEMORY SIZES:")
        print(input_memory_sizes)
        print("*"*100)
        print()
        
        print("WEIGHT DATA:")
        print(weight_data)
        print()
        print("WEIGHT MEMORY SIZES:")
        print(weight_memory_sizes)
        print("*"*100)
        print()
        
        print("OUTPUT DATA:")
        print(output_data)
        print()
        print("OUTPUT MEMORY SIZES:")
        print(output_memory_sizes)
        print("*"*100)
        print()
        
        self.input_data = input_data.tolist()
        self.weight_data = weight_data.tolist()
        self.output_data = output_data.tolist()

        self.input_memory_sizes = input_memory_sizes
        self.weight_memory_sizes = weight_memory_sizes
        self.output_memory_sizes = output_memory_sizes

        self.loop_tiling_lst = loop_tiling_lst
        self.costs = costs
        self.write_backs = write_backs
        self.parallel_for_dims = parallel_for_dims
        self.debug = debug

        # print("Before", self.loop_tiling_lst)
        # if prefetch_factor is not None and prefetch_factor != 0:
        #     self.prefetch_experiment(prefetch_factor)
        # print("After", self.loop_tiling_lst)

    def test_software(self):
        
        input_data = copy.deepcopy(self.input_data)
        weight_data = copy.deepcopy(self.weight_data)
        output_data = copy.deepcopy(self.output_data)
        
        # create the controller
        self.sw_controller = Controller(self.input_memory_sizes[1:], self.weight_memory_sizes[1:], self.output_memory_sizes[1:],
                                        input_data, weight_data, output_data,
                                        self.loop_tiling_lst, self.costs, self.write_backs, self.parallel_for_dims)
        
        self.sw_controller.print_loop()
        
        # run the simulation
        self.sw_controller.run(self.debug)

        # get output of simulation
        controller_output = self.sw_controller.get_base_output()
        
        # compute the true output of the simulation
        true_input_data = copy.deepcopy(self.input_data)
        true_weight_data = np.array(copy.deepcopy(self.weight_data))
        true_output_data = copy.deepcopy(self.output_data)

        for channel in range(len(true_weight_data)):
            for filter in range(len(true_weight_data[channel])):
                for i in range(len(true_output_data[filter])):
                    input_block = np.array(true_input_data[channel][i:i+len(true_weight_data[channel][filter])])
                    true_output_data[filter][i] += np.sum(input_block*true_weight_data[channel][filter])
        
        # compare true output with controller (simulated) output
        diff = true_output_data - controller_output
        if np.sum(diff) == 0:
            print("Passed Correctness Check")
            print("="*100)
            print("="*100)
            print()
            self.sw_controller.print_stats()

            input_memory_trace_queue  = self.sw_controller.input_trace_queue
            weight_memory_trace_queue = self.sw_controller.weight_trace_queue
            output_memory_trace_queue  = self.sw_controller.output_trace_queue
            # print("Input Trace Length:",  len(input_memory_trace_queue))
            # print("Weight Trace Length:", len(weight_memory_trace_queue))
            # print("Output Trace Length:", len(output_memory_trace_queue))
            # print()

        else:
            print("Failed Correctness Check")
            print("="*100)
            print("="*100)
            print()
            print("Expected Output:")
            print(true_output_data)
            print()
            print("Controller Output:")
            print(controller_output)
            print()
            print("Difference:")
            print(diff)

    def test_hardware(self):
        input_memory_trace_queue  = self.sw_controller.input_trace_queue
        weight_memory_trace_queue = self.sw_controller.weight_trace_queue
        output_memory_trace_queue  = self.sw_controller.output_trace_queue

        input_hierarchy_lst  = self.sw_controller.input_memory_hierarchy.hierarchy_lst
        weight_hierarchy_lst = self.sw_controller.weight_memory_hierarchy.hierarchy_lst
        output_hierarchy_lst = self.sw_controller.output_memory_hierarchy.hierarchy_lst

        input_data = copy.deepcopy(self.input_data)
        weight_data = copy.deepcopy(self.weight_data)
        output_data = copy.deepcopy(self.output_data)

        self.hw_controller = Controller_Manager(input_memory_trace_queue, 
                                                weight_memory_trace_queue, 
                                                output_memory_trace_queue,
                                                input_hierarchy_lst, 
                                                weight_hierarchy_lst, 
                                                output_hierarchy_lst,
                                                input_data,
                                                weight_data,
                                                output_data)

        self.hw_controller.run()

    def test(self, full=False):
        self.test_software()
        if not full: return
        self.test_hardware()

        sw_input  = self.sw_controller.get_base_input(flatten=True)
        sw_weight = self.sw_controller.get_base_weight(flatten=True)
        sw_output = self.sw_controller.get_base_output(flatten=True)

        hw_input  = self.hw_controller.get_base_input()
        hw_weight = self.hw_controller.get_base_weight()
        hw_output = self.hw_controller.get_base_output()

        inputs_equal  = np.array_equal(sw_input, hw_input)
        weights_equal = np.array_equal(sw_weight, hw_weight)
        outputs_equal = np.array_equal(sw_output, hw_output)

        if (inputs_equal and weights_equal and outputs_equal):
            print("Passed Hardware Verification")
            print("="*100)
            print("="*100)
            print()
            self.hw_controller.print_stats()
        else:
            print("Failed Hardware Verification")
            if not inputs_equal:
                print("SW Input:")
                print(sw_input)
                print()
                print("HW Input:")
                print(hw_input)
            if not weights_equal:
                print("SW Weight:")
                print(sw_weight)
                print()
                print("HW Weight:")
                print(hw_weight)
            if not outputs_equal:
                print("SW Output:")
                print(sw_output)
                print()
                print("HW Output:")
                print(hw_output)





