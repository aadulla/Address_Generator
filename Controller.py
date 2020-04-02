import numpy as np
import copy
import math
import random

from Hierarchy import Hierarchy
from Memory import InputMemory
from Memory import WeightMemory
from Memory import OutputMemory
from Queue import Trace_Queue

"""
###################################################################################################
LEGEND
---------------------------------------------------------------------------------------------------
1.  op_space_size: the size of the data block a memory prefetches (i.e. the number of elements
    a memory is responsible for at any given time)

2.  memory levels ordering: memories at a lower level are closer to the computation/further
    from the base memory (i.e. memories with smaller op_space_sizes/memories corresponding to the 
    innermost loop counters); memories at a higher level are further from the computation/closer
    to the base memory (i.e. memories with larger op_space_sizes/memories corresponding to the 
    outermost loop counters)
###################################################################################################
"""

""" 
"Controller" is responsible for constructing the memory hierarchy and then simulating it according
to the parameters passed in by the user
"""
class Controller:
    
    """
    "construct_op_spaces" determines the size of the op-space a memory at each level is responsible
    for and returns these sizes as a list
    params:
        tilings:    a list of the loop counters where the first index contains the loop
                    counter at the outer most level and the last index contains the loop counter at 
                    the innermost level
    """
    @staticmethod
    def construct_op_spaces(tilings):
        op_spaces = []
        # since the base memory at the 
        for i in range(1, len(tilings)):
            op_space = 1
            for j in range(i, len(tilings)):
                op_space *= tilings[j]
            op_spaces.append(op_space)
        return op_spaces
    
    """
    "construct_hierarchy_lst" creates the memory hierarchy for a given type by 
    initializing memory instances of the corresponding memory class (InputMemory, WeightMemory, or
    OutputMemory). It returns the hierarchy as a list of memory instances where the first index
    corresponds to the highest level memory (memory right after base memory) and the last index
    corresponds to the lowest level memory (memory which serves computation requests)
    params:
        memory_sizes:   list specifying the size a memory should be initialized with in decreasing
                        order of levels

        op_spaces_dict: mapping between a dimension name and the op_space_size it encapsulates at
                        each level in the hierarchy

        loop_order_lst: 2D list where each 1D list corresponds to a loop block. The first element
                        in the 2D list is a 1D list corresponding to the loop block at the outermost 
                        level; the last elment in the 2D list is a 1D list corresponding to the loop 
                        block at the innermost level. The fist element in the 1D list corresponds to
                        the outermost dimension in that loop block; the last element in the 1D list
                        corresponds to the innermost dimension in that loop block

        costs:          a list specifying the cost of accessing a memory at each level

        write_backs:    a list containing the types that should write back

        memory_type:    the type of memory ("input", "weight", "output")
    """
    @staticmethod
    def construct_hierarchy_lst(memory_sizes, 
                                op_spaces_dict,
                                loop_order_lst,
                                num_levels,
                                trace_queue, 
                                costs, 
                                write_backs, 
                                memory_type):
        hierarchy_lst = []
        for i, memory_size in enumerate(memory_sizes):
            
            # get the order of the loop counter dimensions immediately above the memory's op_space 
            # that determine how the data stored in the memory will change (i.e. how the memory will
            # prefetch data from its parent)
            dim_0_name = loop_order_lst[i][0]
            dim_1_name = loop_order_lst[i][1]
            dim_2_name = loop_order_lst[i][2]
            dim_3_name = loop_order_lst[i][3]
            
            # get the order of the op_space sizes encapsulated by the loop counters immediately 
            # above the memory's op_space that determine how the data stored in the memory will 
            # change (i.e. how the memory will prefetch data from its parent)
            dim_0_name = loop_order_lst[i][0]
            dim_0_size = op_spaces_dict[dim_0_name][i]
            dim_1_size = op_spaces_dict[dim_1_name][i]
            dim_2_size = op_spaces_dict[dim_2_name][i]
            dim_3_size = op_spaces_dict[dim_3_name][i]
            
            # create a mapping between dimension name and dimension op_space size
            upper_dim_map = [{dim_0_name: dim_0_size},
                             {dim_1_name: dim_1_size},
                             {dim_2_name: dim_2_size},
                             {dim_3_name: dim_3_size}]
            
            # check if write back should be enabled in this memory hierarchy
            if memory_type in write_backs:
                should_write_back = True
                write_back_policy = "overwrite"
            else:
                should_write_back = False
                write_back_policy = None

            level = num_levels - i - 1
            
            # initialize input memory hierarchy
            if memory_type == "input":
                op_space_size = op_spaces_dict["channel"][i] * op_spaces_dict["input"][i]
                memory = InputMemory(memory_size, 
                                     op_space_size,
                                     level,
                                     trace_queue, 
                                     upper_dim_map,
                                     costs[i], 
                                     should_write_back, 
                                     write_back_policy)
            
            # initialize weight memory hierarchy
            elif memory_type == "weight":
                op_space_size = op_spaces_dict["channel"][i] * op_spaces_dict["filter"][i] * op_spaces_dict["weight"][i]
                memory = WeightMemory(memory_size, 
                                     op_space_size, 
                                     level,
                                     trace_queue, 
                                     upper_dim_map,
                                     costs[i], 
                                     should_write_back, 
                                     write_back_policy)
                
            # initialize output memory hierarchy
            elif memory_type == "output":
                op_space_size = op_spaces_dict["filter"][i] * op_spaces_dict["output"][i]
                memory = OutputMemory(memory_size, 
                                     op_space_size,
                                     level,
                                     trace_queue,  
                                     upper_dim_map,
                                     costs[i], 
                                     should_write_back, 
                                     write_back_policy)
                
            hierarchy_lst.append(memory)

        return hierarchy_lst
    
    """
    "validate" initially checks all the user parameters passed in to ensure that they are properly
    formatted and are able to create a valid memory hierachy and simulation
    """
    @staticmethod
    def validate(input_memory_sizes, weight_memory_sizes, output_memory_sizes,
                 input_data, weight_data, output_data,
                 tilings_dict, costs):
        
        # check the number of channels matches between weights and inputs
        assert len(weight_data) == len(input_data)
        num_channels = len(weight_data)
        
        # check that each channel has the same number of inputs
        input_lens = []
        for channel in range(len(input_data)):
            input_lens.append(len(input_data[channel]))
        assert np.unique(input_lens).size == 1
        num_inputs = input_lens[0]
        
        # check that the number of weight filters is consistent for each channel
        filter_lens = []
        for channel in range(len(weight_data)):
            filter_lens.append(len(weight_data[channel]))
        assert np.unique(filter_lens).size == 1
        
        # check that each filter has the same number of weights
        weight_lens = []
        for channel in range(len(weight_data)):
            weight_lens = []
            for filter in range(len(weight_data[channel])):
                weight_lens.append(len(weight_data[channel][filter]))
            assert np.unique(weight_lens).size == 1
        num_weights = weight_lens[0]
            
        # check the number of filters matches between weights and outputs
        assert filter_lens[0] == len(output_data)
        num_filters = filter_lens[0]
        
        # check that each filter has the same number of outputs
        output_lens = []
        for filter in range(len(output_data)):
            output_lens.append(len(output_data[filter]))
        assert np.unique(output_lens).size == 1
        num_outputs = output_lens[0]
         
        # check that there are equal levels of tilings across all dimensions
        assert len(tilings_dict["channel"]) == len(tilings_dict["filter"]) == \
               len(tilings_dict["weight"]) == len(tilings_dict["output"])
        
        # check that the tiling matches to the total lens
        assert np.prod(tilings_dict["channel"]) == num_channels
        assert np.prod(tilings_dict["filter"]) == num_filters
        assert np.prod(tilings_dict["weight"]) == num_weights
        assert np.prod(tilings_dict["output"]) == num_outputs
        assert num_inputs == num_weights + num_outputs - 1
        
        # check the number of memory sizes
        assert len(input_memory_sizes) == len(weight_memory_sizes) == \
               len(output_memory_sizes) == len(costs) - 1
        
        # check that one less memory size is specified than tilings (i.e. the user should not 
        # account for initializing the base memory)
        assert len(weight_memory_sizes) == len(tilings_dict["weight"]) - 1
        assert len(output_memory_sizes) == len(tilings_dict["output"]) - 1
        
        # check that input memory sizes can atleast support tile prefetches
        for i, input_memory_size in enumerate(input_memory_sizes):
            assert input_memory_size >= np.prod(tilings_dict["channel"][i+1:]) * \
                                            (np.prod(tilings_dict["weight"][i+1:]) + \
                                             np.prod(tilings_dict["output"][i+1:]) - 1)
           
        # check that weight memory sizes can atleast support tile prefetches 
        for i, weight_memory_size in enumerate(weight_memory_sizes):
            assert weight_memory_size >= np.prod(tilings_dict["channel"][i+1:]) * \
                                         np.prod(tilings_dict["filter"][i+1:]) * \
                                         np.prod(tilings_dict["weight"][i+1:])
            
        # check that output memory sizes can atleast support tile prefetches
        for i, output_memory_size in enumerate(output_memory_sizes):
            assert output_memory_size >= np.prod(tilings_dict["filter"][i+1:]) * \
                                         np.prod(tilings_dict["output"][i+1:])
            
        return num_filters, num_channels, num_inputs, num_weights, num_outputs
    
    def __init__(self, 
                 input_memory_sizes, weight_memory_sizes, output_memory_sizes,
                 input_data, weight_data, output_data,
                 loop_tiling_lst, costs, write_backs):
        
        self.write_backs = write_backs
        self.num_levels = len(input_memory_sizes)
        
        self.loop_order_lst  = []
        self.filter_tilings  = []
        self.channel_tilings = []
        self.weight_tilings  = []
        self.output_tilings  = []
        
        self.tilings_dict = {"filter" : self.filter_tilings,
                             "channel": self.channel_tilings,
                             "weight" : self.weight_tilings,
                             "output" : self.output_tilings}
        
        # create the tilings_dict and loop_order_lst
        for loop_tile in loop_tiling_lst:
            self.loop_order_lst.append([])
            for loop_space in loop_tile:
                loop_key, loop_val = list(loop_space.items())[0]
                self.tilings_dict[loop_key].append(loop_val)
                self.loop_order_lst[-1].append(loop_key)
        
        # validate all user inputs
        num_filters, num_channels, num_inputs, num_weights, num_outputs = Controller.validate(input_memory_sizes, 
                                                                                              weight_memory_sizes, 
                                                                                              output_memory_sizes,
                                                                                              input_data, 
                                                                                              weight_data, 
                                                                                              output_data,
                                                                                              self.tilings_dict, 
                                                                                              costs)
        
        # create op_spaces for the 5 dimensions
        channel_op_spaces = Controller.construct_op_spaces(self.channel_tilings)
        filter_op_spaces  = Controller.construct_op_spaces(self.filter_tilings)
        weight_op_spaces  = Controller.construct_op_spaces(self.weight_tilings)
        output_op_spaces  = Controller.construct_op_spaces(self.output_tilings)
        input_op_spaces   = [x + y - 1 for x, y in zip(weight_op_spaces, output_op_spaces)]
        
        # create op_spaces_dict
        self.op_spaces_dict = {"filter" : filter_op_spaces,
                               "channel": channel_op_spaces,
                               "input"  : input_op_spaces,
                               "weight" : weight_op_spaces,
                               "output" : output_op_spaces}
        
        # create the dim_map for the base memory. the ordering in the list specifies how the memory is laid out
        self.base_dim_map_lst = [{"channel": num_channels},
                                 {"filter" : num_filters},
                                 {"weight" : num_weights},
                                 {"output" : num_outputs}]

        self.base_dim_map_dict = {"channel": num_channels,
                                  "filter" : num_filters,
                                  "weight" : num_weights,
                                  "output" : num_outputs}
        
        # create the base InputMemory
        total_num_inputs = num_inputs * num_channels
        self.input_trace_queue = Trace_Queue()
        # memory args: memory_size, op_space_size, level, trace_queue, upper_dim_map, cost, should_write_back, write_back_policy
        input_base_memory = InputMemory(total_num_inputs, 
                                        total_num_inputs,
                                        self.num_levels, 
                                        self.input_trace_queue,
                                        self.base_dim_map_lst, 
                                        costs[0], 
                                        True, 
                                        "overwrite")         
        input_base_memory.initialize(input_data)
        # create the InputMemory Hierarchy
        input_hierarchy_lst = Controller.construct_hierarchy_lst(input_memory_sizes, 
                                                                 self.op_spaces_dict,
                                                                 self.loop_order_lst,
                                                                 self.num_levels,
                                                                 self.input_trace_queue,
                                                                 costs[1:], 
                                                                 write_backs, 
                                                                 "input") 
        input_hierarchy_lst.insert(0, input_base_memory)
        self.input_memory_hierarchy = Hierarchy(input_hierarchy_lst)
        
        # create the base WeightMemory
        total_num_weights = num_weights * num_channels * num_filters
        self.weight_trace_queue = Trace_Queue()
        # memory args: memory_size, op_space_size, level, trace_queue, upper_dim_map, cost, should_write_back, write_back_policy
        weight_base_memory = WeightMemory(total_num_weights, 
                                          total_num_weights, 
                                          self.num_levels, 
                                          self.weight_trace_queue,
                                          self.base_dim_map_lst, 
                                          costs[0], 
                                          True, 
                                          "overwrite")
        weight_base_memory.initialize(weight_data)
        # create the WeightMemory Hierarchy
        weight_hierarchy_lst = Controller.construct_hierarchy_lst(weight_memory_sizes, 
                                                                 self.op_spaces_dict,
                                                                 self.loop_order_lst,
                                                                 self.num_levels,
                                                                 self.weight_trace_queue,
                                                                 costs[1:],
                                                                 write_backs,
                                                                 "weight") 
        weight_hierarchy_lst.insert(0, weight_base_memory)
        self.weight_memory_hierarchy = Hierarchy(weight_hierarchy_lst)
        
        # create the base OutputMemory
        total_num_outputs = num_outputs * num_filters
        self.output_trace_queue = Trace_Queue()
        # memory args: memory_size, op_space_size, level, trace_queue, upper_dim_map, cost, should_write_back, write_back_policy
        output_base_memory = OutputMemory(total_num_outputs, 
                                          total_num_outputs, 
                                          self.num_levels, 
                                          self.output_trace_queue,
                                          self.base_dim_map_lst, 
                                          costs[0], 
                                          True, 
                                          "overwrite")
        output_base_memory.initialize(output_data)
        # create the base OutputMemory
        output_hierarchy_lst = Controller.construct_hierarchy_lst(output_memory_sizes, 
                                                                 self.op_spaces_dict,
                                                                 self.loop_order_lst,
                                                                 self.num_levels,
                                                                 self.output_trace_queue,
                                                                 costs[1:], 
                                                                 write_backs,
                                                                 "output") 
        output_hierarchy_lst.insert(0, output_base_memory)
        self.output_memory_hierarchy = Hierarchy(output_hierarchy_lst)
        
        # initialize loop_counters and prev_loop_counters
        self.loop_counters = []
        self.prev_loop_counters = []
        for level in self.loop_order_lst:
            self.loop_counters.append({"channel": 0, "filter": 0, "weight": 0, "output": 0})
            self.prev_loop_counters.append({"channel": 0, "filter": 0, "weight": 0, "output": 0})

    def get_weight_prefetch_level(self, loop_order_lst):
        channel_idx = loop_order_lst.index("channel")
        filter_idx  = loop_order_lst.index("filter")
        weight_idx  = loop_order_lst.index("weight")
        return max([channel_idx, filter_idx, weight_idx])

    def get_output_prefetch_level(self, loop_order_lst):
        filter_idx  = loop_order_lst.index("filter")
        output_idx  = loop_order_lst.index("output")
        return max([filter_idx, output_idx])

    def get_first_prefetch_level(self, loop_order_lst):
        weight_prefetch_level = self.get_weight_prefetch_level(loop_order_lst)
        output_prefetch_level = self.get_output_prefetch_level(loop_order_lst)
        return min(weight_prefetch_level, output_prefetch_level)


    """
    "run" performs the entire simulation of the multi-channel/multi-filter convolution
    params:
        debug: indicates whether to simulate in debug mode or not
    """
    def run(self, debug=False):
        
        def recursive_simulate(input_memory_hierarchy, 
                               weight_memory_hierarchy, 
                               output_memory_hierarchy,
                               hierarchy_index, 
                               hierarchy_levels, 
                               loop_order_lst, 
                               tilings_dict,
                               loop_counters,
                               prev_loop_counters,
                               write_backs, debug):
            
            curr_loop_order_lst = self.loop_order_lst[hierarchy_index - 1]
            
            # base case
            if hierarchy_index == hierarchy_levels:
                
                # get the memories at the current level
                input_memory = input_memory_hierarchy[hierarchy_index-1]
                weight_memory = weight_memory_hierarchy[hierarchy_index-1]
                output_memory = output_memory_hierarchy[hierarchy_index-1]

                dim_0 = curr_loop_order_lst[0]
                dim_1 = curr_loop_order_lst[1]
                dim_2 = curr_loop_order_lst[2]
                dim_3 = curr_loop_order_lst[3]

                dim_idxs = {"channel": None, "filter": None, "weight": None, "output": None}

                for idx_0 in range(tilings_dict[dim_0][hierarchy_index - 1]):
                    loop_counters[hierarchy_index-1][dim_0] = idx_0
                    dim_idxs[dim_0] = idx_0

                    for idx_1 in range(tilings_dict[dim_1][hierarchy_index - 1]):
                        loop_counters[hierarchy_index-1][dim_1] = idx_1
                        dim_idxs[dim_1] = idx_1

                        for idx_2 in range(tilings_dict[dim_2][hierarchy_index - 1]):
                            loop_counters[hierarchy_index-1][dim_2] = idx_2
                            dim_idxs[dim_2] = idx_2

                            for idx_3 in range(tilings_dict[dim_3][hierarchy_index - 1]):
                                loop_counters[hierarchy_index-1][dim_3] = idx_3
                                dim_idxs[dim_3] = idx_3

                                channel_idx = dim_idxs["channel"]
                                filter_idx  = dim_idxs["filter"]
                                weight_idx  = dim_idxs["weight"]
                                output_idx  = dim_idxs["output"]

                                # get input data value
                                input_request = {"channel": channel_idx, "input": weight_idx + output_idx}
                                input_val = input_memory.load(input_request)
                                
                                # get weight data value
                                weight_request = {"channel": channel_idx, "filter": filter_idx, "weight": weight_idx}
                                weight_val = weight_memory.load(weight_request)

                                # get output data value
                                output_request = {"filter": filter_idx, "output": output_idx}
                                output_val = output_memory.load(output_request)

                                # barrier synchronization
                                input_memory.barrier_sync()
                                weight_memory.barrier_sync()
                                output_memory.barrier_sync()
                                
                                # if simulating in debug mode, check that the input value and 
                                # weight values match with their global indixes
                                if debug:

                                    global_channel_idx = 0
                                    global_filter_idx = 0
                                    global_weight_idx = 0
                                    global_output_idx = 0

                                    num_channels = self.base_dim_map_dict["channel"]
                                    num_filters = self.base_dim_map_dict["filter"]
                                    num_weights = self.base_dim_map_dict["weight"]
                                    num_outputs = self.base_dim_map_dict["output"]
                                    num_inputs = num_weights + num_outputs - 1

                                    for i, loop_counter in enumerate(loop_counters):

                                        channel_total = 1
                                        for tiling in tilings_dict["channel"][i+1:]:
                                            channel_total *= tiling
                                        global_channel_idx += loop_counter["channel"]*channel_total

                                        filter_total = 1
                                        for tiling in tilings_dict["filter"][i+1:]:
                                            filter_total *= tiling
                                        global_filter_idx += loop_counter["filter"]*filter_total

                                        weight_total = 1
                                        for tiling in tilings_dict["weight"][i+1:]:
                                            weight_total *= tiling
                                        global_weight_idx += loop_counter["weight"]*weight_total

                                        output_total = 1
                                        for tiling in tilings_dict["output"][i+1:]:
                                            output_total *= tiling
                                        global_output_idx += loop_counter["output"]*output_total

                                    global_weight_val = global_channel_idx * num_filters * num_weights + \
                                                        global_filter_idx * num_weights + \
                                                        global_weight_idx

                                    global_output_val = global_filter_idx * num_outputs + \
                                                        global_output_idx

                                    global_input_val  = global_channel_idx * num_inputs + \
                                                        global_weight_idx + global_output_idx

                                    if (input_val != global_input_val):
                                        print("INPUT VALUE MISMATCH")
                                        print(loop_counters)
                                        print("Expected: ", global_input_val)
                                        print("Returned: ", input_val)
                                        input_memory.print()
                                        assert False
                                    if (weight_val != global_weight_val):
                                        print("WEIGHT VALUE MISMATCH")
                                        print(loop_counters)
                                        print("Expected: ", global_weight_val)
                                        print("Returned: ", weight_val)
                                        weight_memory.print()
                                        assert False
                                    
                                # compute new output data value and store it back into output memory
                                output_val += input_val * weight_val
                                # print(output_val)
                                output_memory.store(output_request, output_val)

                                # barrier synchronization
                                input_memory.barrier_sync()
                                weight_memory.barrier_sync()
                                output_memory.barrier_sync()
                            
                return
            
            # recursive case
            else:
                
                # get the memories at the current level
                input_memory = input_memory_hierarchy[hierarchy_index]
                weight_memory = weight_memory_hierarchy[hierarchy_index]
                output_memory = output_memory_hierarchy[hierarchy_index]
                
                input_prev_block_start = 0
                input_curr_block_start = 0
                input_delta = 0
                old_prev_loop_counters = None

                dim_0 = curr_loop_order_lst[0]
                dim_1 = curr_loop_order_lst[1]
                dim_2 = curr_loop_order_lst[2]
                dim_3 = curr_loop_order_lst[3]

                dim_idxs   = {"channel": None, "filter": None, "weight": None, "output": None}
                dim_totals = {"channel": None, "filter": None, "weight": None, "output": None}

                input_loop_counters = {"channel": 0, "filter": 0, "weight": 0, "output": 0}
                weight_loop_counters = {"channel": 0, "filter": 0, "weight": 0, "output": 0}
                output_loop_counters = {"channel": 0, "filter": 0, "weight": 0, "output": 0}

                start = False
                first_prefetch_level  = self.get_first_prefetch_level(curr_loop_order_lst)
                weight_prefetch_level = self.get_weight_prefetch_level(curr_loop_order_lst)
                output_prefetch_level = self.get_output_prefetch_level(curr_loop_order_lst)

                curr_level = 0
                if curr_level == first_prefetch_level: start = True
                
                loop_counters[hierarchy_index-1][dim_0] = 0
                loop_counters[hierarchy_index-1][dim_1] = 0
                loop_counters[hierarchy_index-1][dim_2] = 0
                loop_counters[hierarchy_index-1][dim_3] = 0

                for idx_0 in range(tilings_dict[dim_0][hierarchy_index - 1]):
                    dim_idxs[dim_0] = idx_0
                    dim_totals[dim_0] = 1
                    for tile_0 in tilings_dict[dim_0][hierarchy_index:]:
                        dim_totals[dim_0] *= tile_0
                    loop_counters[hierarchy_index-1][dim_0] = idx_0

                    if weight_prefetch_level == 0:
                        # prefetch data for weight memory
                        weight_memory.prefetch(None, 
                                               loop_counters[hierarchy_index-1], 
                                               weight_loop_counters, 
                                               curr_loop_order_lst)
                        weight_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                    if output_prefetch_level == 0:
                        # prefetch data for output memory
                        output_memory.prefetch(None, 
                                               loop_counters[hierarchy_index-1], 
                                               output_loop_counters, 
                                               curr_loop_order_lst)
                        output_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                    curr_level = 1
                    if curr_level == first_prefetch_level: start = True
                    
                    loop_counters[hierarchy_index-1][dim_1] = 0
                    loop_counters[hierarchy_index-1][dim_2] = 0
                    loop_counters[hierarchy_index-1][dim_3] = 0

                    for idx_1 in range(tilings_dict[dim_1][hierarchy_index - 1]):
                        dim_idxs[dim_1] = idx_1
                        dim_totals[dim_1] = 1
                        for tile_1 in tilings_dict[dim_1][hierarchy_index:]:
                            dim_totals[dim_1] *= tile_1
                        loop_counters[hierarchy_index-1][dim_1] = idx_1

                        if weight_prefetch_level == 1:
                            # prefetch data for weight memory
                            weight_memory.prefetch(None, 
                                                   loop_counters[hierarchy_index-1], 
                                                   weight_loop_counters, 
                                                   curr_loop_order_lst)
                            weight_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                        if output_prefetch_level == 1:
                            # prefetch data for output memory
                            output_memory.prefetch(None, 
                                                   loop_counters[hierarchy_index-1], 
                                                   output_loop_counters, 
                                                   curr_loop_order_lst)
                            output_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])
                        
                        curr_level = 2
                        if curr_level == first_prefetch_level: start = True

                        loop_counters[hierarchy_index-1][dim_2] = 0
                        loop_counters[hierarchy_index-1][dim_3] = 0
                        
                        for idx_2 in range(tilings_dict[dim_2][hierarchy_index - 1]):
                            dim_idxs[dim_2] = idx_2
                            dim_totals[dim_2] = 1
                            for tile_2 in tilings_dict[dim_2][hierarchy_index:]:
                                dim_totals[dim_2] *= tile_2
                            loop_counters[hierarchy_index-1][dim_2] = idx_2

                            if weight_prefetch_level == 2:
                                # prefetch data for weight memory
                                weight_memory.prefetch(None, 
                                                       loop_counters[hierarchy_index-1], 
                                                       weight_loop_counters, 
                                                       curr_loop_order_lst)
                                weight_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                            if output_prefetch_level == 2:
                                # prefetch data for output memory
                                output_memory.prefetch(None, 
                                                       loop_counters[hierarchy_index-1], 
                                                       output_loop_counters, 
                                                       curr_loop_order_lst)
                                output_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                            curr_level = 3
                            if curr_level == first_prefetch_level: start = True

                            loop_counters[hierarchy_index-1][dim_3] = 0

                            for idx_3 in range(tilings_dict[dim_3][hierarchy_index - 1]):
                                dim_idxs[dim_3] = idx_3
                                dim_totals[dim_3] = 1
                                for tile_3 in tilings_dict[dim_3][hierarchy_index:]:
                                    dim_totals[dim_3] *= tile_3
                                loop_counters[hierarchy_index-1][dim_3] = idx_3

                                if weight_prefetch_level == 3:
                                    # prefetch data for weight memory
                                    weight_memory.prefetch(None, 
                                                           loop_counters[hierarchy_index-1], 
                                                           weight_loop_counters, 
                                                           curr_loop_order_lst)
                                    weight_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                                if output_prefetch_level == 3:
                                    # prefetch data for output memory
                                    output_memory.prefetch(None, 
                                                           loop_counters[hierarchy_index-1], 
                                                           output_loop_counters, 
                                                           curr_loop_order_lst)
                                    output_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                                channel_idx = dim_idxs["channel"]
                                filter_idx  = dim_idxs["filter"]
                                weight_idx  = dim_idxs["weight"]
                                output_idx  = dim_idxs["output"]
                                
                                channel_total = dim_totals["channel"]
                                filter_total  = dim_totals["filter"]
                                weight_total  = dim_totals["weight"]
                                output_total  = dim_totals["output"]

                                # calculate input delta
                                input_curr_block_start = (weight_idx*weight_total + output_idx*output_total)
                                input_delta = input_curr_block_start - input_prev_block_start

                                # check if this is the first time entering this loop block
                                if start:
                                    # preform a fresh prefetch
                                    is_fresh_prefetch = input_memory.prefetch(None, 
                                                          loop_counters[hierarchy_index-1], 
                                                          input_loop_counters, 
                                                          curr_loop_order_lst)
    
                                else:
                                    # perform a delta prefetch
                                    is_fresh_prefetch = input_memory.prefetch(input_delta, 
                                                          loop_counters[hierarchy_index-1], 
                                                          input_loop_counters, 
                                                          curr_loop_order_lst)

                                if is_fresh_prefetch: input_loop_counters = copy.deepcopy(loop_counters[hierarchy_index-1])

                                # simulate next level 
                                recursive_simulate(input_memory_hierarchy, 
                                                   weight_memory_hierarchy, 
                                                   output_memory_hierarchy,
                                                   hierarchy_index+1, 
                                                   hierarchy_levels, 
                                                   loop_order_lst[1:], 
                                                   tilings_dict,
                                                   loop_counters,
                                                   prev_loop_counters,
                                                   write_backs,
                                                   debug)

                                input_prev_block_start = input_curr_block_start
                                old_prev_loop_counters = copy.deepcopy(prev_loop_counters)
                                prev_loop_counters = copy.deepcopy(loop_counters)
                                start = False
                
                # perform write backs 
                if "input" in write_backs:
                    input_memory.write_back_op_space(input_loop_counters)
                if "weight" in write_backs:
                    weight_memory.write_back_op_space(weight_loop_counters)
                if "output" in write_backs:
                    output_memory.write_back_op_space(output_loop_counters)
                    
                return
            
        # start the simulation
        hierarchy_index = 1
        hierarchy_levels = len(self.input_memory_hierarchy)
        recursive_simulate(self.input_memory_hierarchy, 
                           self.weight_memory_hierarchy,
                           self.output_memory_hierarchy,
                           hierarchy_index,
                           hierarchy_levels,
                           self.loop_order_lst,
                           self.tilings_dict,
                           self.loop_counters, 
                           self.prev_loop_counters,
                           self.write_backs, debug)

        input_memory  = self.input_memory_hierarchy[0]
        weight_memory = self.weight_memory_hierarchy[0]
        output_memory = self.output_memory_hierarchy[0]

        input_memory.end_trace()
        weight_memory.end_trace()
        output_memory.end_trace()


    """
    "get_base_input" formats the flattened 1D memory array at the base input memory into a 2D
    numpy array
    """
    def get_base_input(self, flatten=False):
        flat_input_lst = self.input_memory_hierarchy[0].memory_array
        if flatten: return flat_input_lst

        num_inputs = self.base_dim_map_dict["weight"] + self.base_dim_map_dict["output"] - 1
        num_channels = self.base_dim_map_dict["channel"]
        
        return np.reshape(flat_input_lst, (num_channels, num_inputs))

    """
    "get_base_weight" formats the flattened 1D memory array at the base weight memory into a 2D
    numpy array
    """
    def get_base_weight(self, flatten=False):
        flat_weight_lst = self.weight_memory_hierarchy[0].memory_array
        if flatten: return flat_weight_lst

        num_weights = self.base_dim_map_dict["weight"]
        num_channels = self.base_dim_map_dict["channel"]
        num_filters = self.base_dim_map_dict["filter"]
        
        return np.reshape(flat_weight_lst, (num_channels, num_filters, num_weights))

    """
    "get_base_output" formats the flattened 1D memory array at the base output memory into a 2D
    numpy array
    """
    def get_base_output(self, flatten=False):
        flat_output_lst = self.output_memory_hierarchy[0].memory_array
        if flatten: return flat_output_lst

        num_outputs = self.base_dim_map_dict["output"]
        num_filters = self.base_dim_map_dict["filter"]
        
        return np.reshape(flat_output_lst, (num_filters, num_outputs))
            
    
    def print_loop(self):
        print("LOOP HIERARCHY")
        level = -1
        for i, block in enumerate(self.loop_order_lst):
            for dim in block:
                level += 1
                print("  " * level, end="")
                print(dim + ": " + str(self.tilings_dict[dim][i]))
        print("*"*100)
        print()
        
    def print(self):
        print("Simulated Looping:")
        self.print_loop()
        print()
        print("INPUT MEMORY HIERARCHY")
        self.input_memory_hierarchy.print()
        print("*"*100)
        print()
        print("WEIGHT MEMORY HIERARCHY")
        self.weight_memory_hierarchy.print()
        print("*"*100)
        print()
        print("OUTPUT MEMORY HIERARCHY")
        self.output_memory_hierarchy.print()
        print("*"*100)
        print()
        
    def print_stats(self):
        print("INPUT MEMORY HIERARCHY")
        print("*"*100)
        self.input_memory_hierarchy.print_stats()
        print()

        print("WEIGHT MEMORY HIERARCHY")
        print("*"*100)
        self.weight_memory_hierarchy.print_stats()
        print()

        print("OUTPUT MEMORY HIERARCHY")
        print("*"*100)
        self.output_memory_hierarchy.print_stats()
        print()