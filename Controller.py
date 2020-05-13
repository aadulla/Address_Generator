import numpy as np
import copy
import math
import random
import threading

from Hierarchy import Hierarchy
from RegisterFile import RegisterFile
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
                write_back_policy = "overwrite"

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
                 loop_tiling_lst, costs, write_backs, parallel_for_dims):
        
        self.write_backs = write_backs
        self.parallel_for_dims = parallel_for_dims
        # add 1 to account for not including base memory size
        self.num_levels = len(input_memory_sizes) + 1
        
        self.loop_tiling_lst = loop_tiling_lst

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
        for loop_tile in self.loop_tiling_lst:
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
                                        self.num_levels-1, 
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
                                                                 self.num_levels-1,
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
                                          self.num_levels-1, 
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
                                                                 self.num_levels-1,
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
                                          self.num_levels-1, 
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
                                                                 self.num_levels-1,
                                                                 self.output_trace_queue,
                                                                 costs[1:], 
                                                                 write_backs,
                                                                 "output") 
        output_hierarchy_lst.insert(0, output_base_memory)
        self.output_memory_hierarchy = Hierarchy(output_hierarchy_lst)

        # change L1 output memory to increment policy so can captures delta from L0 computations
        self.output_memory_hierarchy[-2].write_back_policy = "increment"
        
        # initialize loop_counters and prev_loop_counters
        self.loop_counters = []
        # self.prev_loop_counters = []
        for level in self.loop_order_lst:
            self.loop_counters.append({"channel": 0, "filter": 0, "weight": 0, "output": 0})
            # self.prev_loop_counters.append({"channel": 0, "filter": 0, "weight": 0, "output": 0})

        # initialize parallel unrolling
        self.initialize_parallel_unrolling()

        self.memory_stats = {"input": [], "weight": [], "output": []}

    def initialize_parallel_unrolling(self):
        # check if we should do parallel unrolling
        if self.parallel_for_dims is None: return

        # get number of PEs in PE array
        # parallel unroll the second to last tile because each PE has its own RFile to do the last tile
        self.num_PEs = 1
        for dim in self.parallel_for_dims:
            L1_tile_size = self.tilings_dict[dim][-2]
            self.num_PEs *= L1_tile_size

        # create loop counter assignment for each PE
        # assign specific iteration to each PE based on second to last tile dim indices
        PE_loop_counter_dict = dict()
        for dim in self.parallel_for_dims:
            PE_loop_counter_dict.update({dim: None})

        self.loop_counter_asst_lst = []
        for i in range(self.num_PEs):
            self.loop_counter_asst_lst.append(copy.deepcopy(PE_loop_counter_dict))

        stride = 1
        for dim in reversed(self.parallel_for_dims):
            L1_tile_size = self.tilings_dict[dim][-2]
            for i in range((self.num_PEs//L1_tile_size)//stride):
                for j in range(L1_tile_size):
                    for k in range(stride):
                        self.loop_counter_asst_lst[i*L1_tile_size*stride + j*stride + k][dim] = j
            stride *= L1_tile_size

        # e.g 
        # parallel_for_dims: [r, c]
        # L1_tile_sizes: [r=2, c=3]

        # before:
        # (r: None, c: None), (r: None, c: None), (r: None, c: None)
        # (r: None, c: None), (r: None, c: None), (r: None, c: None)

        # after:
        # (r: 0, c: 0), (r: 0, c: 1), (r: 0, c: 2)
        # (r: 1, c: 0), (r: 1, c: 1), (r: 1, c: 2)

        # unroll L0 memories in hierarchies
        self.input_memory_hierarchy.parallel_unroll(self.num_PEs)
        self.weight_memory_hierarchy.parallel_unroll(self.num_PEs)
        self.output_memory_hierarchy.parallel_unroll(self.num_PEs)

        # create RFiles from input, weight, and output memories
        self.RFiles = []
        for i in range(self.num_PEs):
            # get L0 memories contained this RFile
            L0_input_memory  = self.input_memory_hierarchy[-1][i]
            L0_weight_memory = self.weight_memory_hierarchy[-1][i]
            L0_output_memory = self.output_memory_hierarchy[-1][i]

            # get loop counter assignment for this RFile
            loop_counter_asst = self.loop_counter_asst_lst[i]

            # create RFile
            RFile = RegisterFile(loop_counter_asst,
                                 L0_input_memory, 
                                 L0_weight_memory, 
                                 L0_output_memory)
            self.RFiles.append(RFile)

    def log_memory_stats(self):

        def get_hierarchy_stats(memory_hierarchy):
            hierarchy_stats = []
            for memory in memory_hierarchy:
                read_count = 0
                write_count = 0
                if type(memory) == list:
                    for parallel_memory in memory:
                        read_count += parallel_memory.read_count
                        write_count += parallel_memory.write_count
                else:
                    read_count = memory.read_count
                    write_count = memory.write_count
                hierarchy_stats.append({"read count": read_count, "write_count": write_count})
            return hierarchy_stats

        input_hierarchy_stats  = get_hierarchy_stats(self.input_memory_hierarchy)
        weight_hierarchy_stats = get_hierarchy_stats(self.weight_memory_hierarchy)
        output_hierarchy_stats = get_hierarchy_stats(self.output_memory_hierarchy)

        self.memory_stats["input"].append(input_hierarchy_stats)
        self.memory_stats["weight"].append(weight_hierarchy_stats)
        self.memory_stats["output"].append(output_hierarchy_stats)

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

    def memory_prefetch_wrapper(self,
                                memory,
                                hierarchy_index,
                                curr_loop_counters, 
                                prev_loop_counters,
                                loop_order_lst):

        if prev_loop_counters is None:
            # if memory.memory_type == "output": print("prefetch output")
            # if memory.memory_type == "weight": print("prefetch weight")
            memory.prefetch(None, curr_loop_counters[hierarchy_index], loop_order_lst)

        else:
            for dim in memory.dependency_set:
                if curr_loop_counters[hierarchy_index][dim] != prev_loop_counters[hierarchy_index][dim]:
                    # if memory.memory_type == "output": print("prefetch output")
                    # if memory.memory_type == "weight": print("prefetch weight")
                    memory.prefetch(None, curr_loop_counters[hierarchy_index], loop_order_lst)
                    break

    def Ln_recursive_simulate(self, hierarchy_index, debug):

        curr_loop_order_lst = self.loop_order_lst[hierarchy_index - 1]
        prev_loop_counters = None

        # get the memories at the current level
        input_memory  = self.input_memory_hierarchy[hierarchy_index]
        weight_memory = self.weight_memory_hierarchy[hierarchy_index]
        output_memory = self.output_memory_hierarchy[hierarchy_index]
      
        input_prev_block_start = 0
        input_curr_block_start = 0
        input_delta = 0

        dim_0 = curr_loop_order_lst[0]
        dim_1 = curr_loop_order_lst[1]
        dim_2 = curr_loop_order_lst[2]
        dim_3 = curr_loop_order_lst[3]

        dim_idxs   = {"channel": None, "filter": None, "weight": None, "output": None}
        dim_totals = {"channel": None, "filter": None, "weight": None, "output": None}

        start = False
        first_prefetch_level  = self.get_first_prefetch_level(curr_loop_order_lst)

        curr_level = 0
        if curr_level == first_prefetch_level: start = True

        self.loop_counters[hierarchy_index-1][dim_0] = 0
        self.loop_counters[hierarchy_index-1][dim_1] = 0
        self.loop_counters[hierarchy_index-1][dim_2] = 0
        self.loop_counters[hierarchy_index-1][dim_3] = 0

        for idx_0 in range(self.tilings_dict[dim_0][hierarchy_index - 1]):
            dim_idxs[dim_0] = idx_0
            dim_totals[dim_0] = 1
            for tile_0 in self.tilings_dict[dim_0][hierarchy_index:]:
                dim_totals[dim_0] *= tile_0
            self.loop_counters[hierarchy_index-1][dim_0] = idx_0

            curr_level = 1
            if curr_level == first_prefetch_level: start = True

            for idx_1 in range(self.tilings_dict[dim_1][hierarchy_index - 1]):
                dim_idxs[dim_1] = idx_1
                dim_totals[dim_1] = 1
                for tile_1 in self.tilings_dict[dim_1][hierarchy_index:]:
                    dim_totals[dim_1] *= tile_1
                self.loop_counters[hierarchy_index-1][dim_1] = idx_1
              
                curr_level = 2
                if curr_level == first_prefetch_level: start = True

                for idx_2 in range(self.tilings_dict[dim_2][hierarchy_index - 1]):
                    dim_idxs[dim_2] = idx_2
                    dim_totals[dim_2] = 1
                    for tile_2 in self.tilings_dict[dim_2][hierarchy_index:]:
                        dim_totals[dim_2] *= tile_2
                    self.loop_counters[hierarchy_index-1][dim_2] = idx_2

                    curr_level = 3
                    if curr_level == first_prefetch_level: start = True

                    for idx_3 in range(self.tilings_dict[dim_3][hierarchy_index - 1]):
                        dim_idxs[dim_3] = idx_3
                        dim_totals[dim_3] = 1
                        for tile_3 in self.tilings_dict[dim_3][hierarchy_index:]:
                            dim_totals[dim_3] *= tile_3
                        self.loop_counters[hierarchy_index-1][dim_3] = idx_3

                        channel_idx = dim_idxs["channel"]
                        filter_idx  = dim_idxs["filter"]
                        weight_idx  = dim_idxs["weight"]
                        output_idx  = dim_idxs["output"]
                      
                        channel_total = dim_totals["channel"]
                        filter_total  = dim_totals["filter"]
                        weight_total  = dim_totals["weight"]
                        output_total  = dim_totals["output"]

                        # prefetch weight and output memories
                        self.memory_prefetch_wrapper(weight_memory, 
                                                     hierarchy_index-1,
                                                     self.loop_counters,
                                                     prev_loop_counters,
                                                     curr_loop_order_lst)

                        self.memory_prefetch_wrapper(output_memory, 
                                                     hierarchy_index-1,
                                                     self.loop_counters,
                                                     prev_loop_counters,
                                                     curr_loop_order_lst)

                        # calculate input delta
                        if (input_memory.get_lowest_dim() == "input"):
                            input_curr_block_start = (weight_idx*weight_total + output_idx*output_total)
                        else:
                            input_curr_block_start = (channel_idx*channel_total)
                        input_delta = input_curr_block_start - input_prev_block_start

                        # check if this is the first time entering this loop block
                        if start:
                            # print("prefetch input fresh")
                            # preform a fresh prefetch
                            is_fresh_prefetch = input_memory.prefetch(None, 
                                                                      self.loop_counters[hierarchy_index-1],
                                                                      curr_loop_order_lst)

                        else:
                            # print("prefetch input delta", input_delta)
                            # perform a delta prefetch
                            is_fresh_prefetch = input_memory.prefetch(input_delta, 
                                                                      self.loop_counters[hierarchy_index-1],
                                                                      curr_loop_order_lst)

                        # check if next level is L1 and we are doing a parallel unroll
                        if (hierarchy_index+1 == self.num_levels-1) and (self.parallel_for_dims is not None):
                            self.L0_parallel_simulate(hierarchy_index+1, debug)

                        # check if next level is L0
                        elif (hierarchy_index+1 == self.num_levels):
                            L0_input_memory = self.input_memory_hierarchy[-1]
                            L0_weight_memory = self.weight_memory_hierarchy[-1]
                            L0_output_memory = self.output_memory_hierarchy[-1]
                            self.L0_compute(hierarchy_index+1, debug, self.loop_counters,
                                            L0_input_memory, L0_weight_memory, L0_output_memory)

                        # just do normal level recursion
                        else:
                            self.Ln_recursive_simulate(hierarchy_index+1, debug)

                        input_prev_block_start = input_curr_block_start
                        start = False

                        prev_loop_counters = copy.deepcopy(self.loop_counters)

            if hierarchy_index == 1: self.log_memory_stats()
      
        # perform write backs 
        if "input" in self.write_backs:
            input_memory.write_back_op_space()
        if "weight" in self.write_backs:
            weight_memory.write_back_op_space()
        if "output" in self.write_backs:
            output_memory.write_back_op_space()

          
        return

    def RFile_simulate(self, 
                       RFile, 
                       start,
                       debug,
                       hierarchy_index,
                       prev_loop_counters,
                       curr_loop_order_lst,
                       channel_total,
                       filter_total,
                       weight_total,
                       output_total):

        # overwrite loop counters with RFile assigned loop counter
        RFile_curr_loop_counters = copy.deepcopy(self.loop_counters)
        RFile_curr_loop_counters[hierarchy_index-1].update(RFile.loop_counter_asst)

        if prev_loop_counters is not None:
            RFile_prev_loop_counters = copy.deepcopy(prev_loop_counters)
            RFile_prev_loop_counters[hierarchy_index-1].update(RFile.loop_counter_asst)
        else:
            RFile_prev_loop_counters = prev_loop_counters

        channel_idx = RFile_curr_loop_counters[hierarchy_index-1]["channel"]
        filter_idx  = RFile_curr_loop_counters[hierarchy_index-1]["filter"]
        weight_idx  = RFile_curr_loop_counters[hierarchy_index-1]["weight"]
        output_idx  = RFile_curr_loop_counters[hierarchy_index-1]["output"]

        # prefetch weight and output memories
        self.memory_prefetch_wrapper(RFile.weight_memory, 
                                     hierarchy_index-1,
                                     RFile_curr_loop_counters,
                                     RFile_prev_loop_counters,
                                     curr_loop_order_lst)

        self.memory_prefetch_wrapper(RFile.output_memory, 
                                     hierarchy_index-1,
                                     RFile_curr_loop_counters,
                                     RFile_prev_loop_counters,
                                     curr_loop_order_lst)

        # calculate input delta
        RFile.input_curr_block_start = (weight_idx*weight_total + output_idx*output_total)
        input_delta = RFile.input_curr_block_start - RFile.input_prev_block_start

        # check if this is the first time entering this loop block
        if start:
            # preform a fresh prefetch
            is_fresh_prefetch = RFile.input_memory.prefetch(None, 
                                                            RFile_curr_loop_counters[hierarchy_index-1],
                                                            curr_loop_order_lst)

        else:
            # perform a delta prefetch
            is_fresh_prefetch = RFile.input_memory.prefetch(input_delta, 
                                                            RFile_curr_loop_counters[hierarchy_index-1],
                                                            curr_loop_order_lst)

        self.L0_compute(hierarchy_index+1, debug, RFile_curr_loop_counters,
                        RFile.input_memory, RFile.weight_memory, RFile.output_memory)

        RFile.input_prev_block_start = RFile.input_curr_block_start

    def L0_parallel_simulate(self, hierarchy_index, debug):

        curr_loop_order_lst = self.loop_order_lst[hierarchy_index - 1]
        prev_loop_counters = None

        dim_0 = curr_loop_order_lst[0]
        dim_1 = curr_loop_order_lst[1]
        dim_2 = curr_loop_order_lst[2]
        dim_3 = curr_loop_order_lst[3]

        dim_idxs   = {"channel": None, "filter": None, "weight": None, "output": None}
        dim_totals = {"channel": None, "filter": None, "weight": None, "output": None}

        start = False
        first_prefetch_level  = self.get_first_prefetch_level(curr_loop_order_lst)
      
        self.loop_counters[hierarchy_index-1][dim_0] = 0
        self.loop_counters[hierarchy_index-1][dim_1] = 0
        self.loop_counters[hierarchy_index-1][dim_2] = 0
        self.loop_counters[hierarchy_index-1][dim_3] = 0

        curr_level = 0
        if curr_level == first_prefetch_level: start = True

        if 4-curr_level <= len(self.parallel_for_dims): num_iters_dim_0 = 1
        else: num_iters_dim_0 = self.tilings_dict[dim_0][hierarchy_index - 1]

        for idx_0 in range(num_iters_dim_0):
            dim_idxs[dim_0] = idx_0
            dim_totals[dim_0] = 1
            for tile_0 in self.tilings_dict[dim_0][hierarchy_index:]:
                dim_totals[dim_0] *= tile_0
            self.loop_counters[hierarchy_index-1][dim_0] = idx_0

            curr_level = 1
            if curr_level == first_prefetch_level: start = True
          
            if 4-curr_level <= len(self.parallel_for_dims): num_iters_dim_1 = 1
            else: num_iters_dim_1 = self.tilings_dict[dim_1][hierarchy_index - 1]

            for idx_1 in range(num_iters_dim_1):
                dim_idxs[dim_1] = idx_1
                dim_totals[dim_1] = 1
                for tile_1 in self.tilings_dict[dim_1][hierarchy_index:]:
                    dim_totals[dim_1] *= tile_1
                self.loop_counters[hierarchy_index-1][dim_1] = idx_1
              
                curr_level = 2
                if curr_level == first_prefetch_level: start = True
              
                if 4-curr_level <= len(self.parallel_for_dims): num_iters_dim_2 = 1
                else: num_iters_dim_2 = self.tilings_dict[dim_2][hierarchy_index - 1]

                for idx_2 in range(num_iters_dim_2):
                    dim_idxs[dim_2] = idx_2
                    dim_totals[dim_2] = 1
                    for tile_2 in self.tilings_dict[dim_2][hierarchy_index:]:
                        dim_totals[dim_2] *= tile_2
                    self.loop_counters[hierarchy_index-1][dim_2] = idx_2

                    curr_level = 3
                    if curr_level == first_prefetch_level: start = True
                  
                    if 4-curr_level <= len(self.parallel_for_dims): num_iters_dim_3 = 1
                    else: num_iters_dim_3 = self.tilings_dict[dim_3][hierarchy_index - 1]

                    for idx_3 in range(num_iters_dim_3):
                        dim_idxs[dim_3] = idx_3
                        dim_totals[dim_3] = 1
                        for tile_3 in self.tilings_dict[dim_3][hierarchy_index:]:
                            dim_totals[dim_3] *= tile_3
                        self.loop_counters[hierarchy_index-1][dim_3] = idx_3
                      
                        channel_total = dim_totals["channel"]
                        filter_total  = dim_totals["filter"]
                        weight_total  = dim_totals["weight"]
                        output_total  = dim_totals["output"]

                        # parallel unroll all RFiles
                        RFile_threads = []
                        for RFile in self.RFiles:
                            RFile_threads.append(threading.Thread(target=self.RFile_simulate, 
                                                                  args=(RFile, 
                                                                        start,
                                                                        debug,
                                                                        hierarchy_index,
                                                                        prev_loop_counters,
                                                                        curr_loop_order_lst,
                                                                        channel_total,
                                                                        filter_total,
                                                                        weight_total,
                                                                        output_total)))
                        for RFile_thread in RFile_threads:
                            RFile_thread.start()
                        for RFile_thread in RFile_threads:
                            RFile_thread.join()

                    start = False
                    prev_loop_counters = copy.deepcopy(self.loop_counters)

            if hierarchy_index == 1: self.log_memory_stats()

        # parallel unroll all RFiles
        for RFile in self.RFiles:
            # perform write backs 
            if "input" in self.write_backs:
                RFile.input_memory.write_back_op_space()
            if "weight" in self.write_backs:
                RFile.weight_memory.write_back_op_space()
            if "output" in self.write_backs:
                RFile.output_memory.write_back_op_space()

          
        return

    def L0_compute(self, hierarchy_index, debug, loop_counters,
                   input_memory, weight_memory, output_memory):

        curr_loop_order_lst = self.loop_order_lst[hierarchy_index - 1]

        dim_0 = curr_loop_order_lst[0]
        dim_1 = curr_loop_order_lst[1]
        dim_2 = curr_loop_order_lst[2]
        dim_3 = curr_loop_order_lst[3]

        dim_idxs = {"channel": None, "filter": None, "weight": None, "output": None}

        for idx_0 in range(self.tilings_dict[dim_0][hierarchy_index - 1]):
            loop_counters[hierarchy_index-1][dim_0] = idx_0
            dim_idxs[dim_0] = idx_0

            for idx_1 in range(self.tilings_dict[dim_1][hierarchy_index - 1]):
                loop_counters[hierarchy_index-1][dim_1] = idx_1
                dim_idxs[dim_1] = idx_1

                for idx_2 in range(self.tilings_dict[dim_2][hierarchy_index - 1]):
                    loop_counters[hierarchy_index-1][dim_2] = idx_2
                    dim_idxs[dim_2] = idx_2

                    for idx_3 in range(self.tilings_dict[dim_3][hierarchy_index - 1]):
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
                      
                        # if simulating in debug mode, check that the input values and 
                        # weight values match with their global indices
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
                                for tiling in self.tilings_dict["channel"][i+1:]:
                                    channel_total *= tiling
                                global_channel_idx += loop_counter["channel"]*channel_total

                                filter_total = 1
                                for tiling in self.tilings_dict["filter"][i+1:]:
                                    filter_total *= tiling
                                global_filter_idx += loop_counter["filter"]*filter_total

                                weight_total = 1
                                for tiling in self.tilings_dict["weight"][i+1:]:
                                    weight_total *= tiling
                                global_weight_idx += loop_counter["weight"]*weight_total

                                output_total = 1
                                for tiling in self.tilings_dict["output"][i+1:]:
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
                        output_memory.store(output_request, output_val)

                        # barrier synchronization
                        input_memory.barrier_sync()
                        weight_memory.barrier_sync()
                        output_memory.barrier_sync()

        return



    """
    "run" performs the entire simulation of the multi-channel/multi-filter convolution
    params:
        debug: indicates whether to simulate in debug mode or not
    """
    def run(self, debug=False):
        
        """
        ############################
        """

        # start the simulation
        hierarchy_index = 1
        if (self.num_levels <= 2) and (self.parallel_for_dims is not None):
            self.L0_parallel_simulate(hierarchy_index, debug)
        else:
            self.Ln_recursive_simulate(hierarchy_index, debug)

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

        for tile in self.loop_tiling_lst:
            print(tile)
        print()

        # print("Input Memory Iteration Stats")
        # for iteration in self.memory_stats["input"]:
        #     print(iteration)
        # print()

        # print("Weight Memory Iteration Stats")
        # for iteration in self.memory_stats["weight"]:
        #     print(iteration)
        # print()

        # print("Output Memory Iteration Stats")
        # for iteration in self.memory_stats["output"]:
        #     print(iteration)
        # print()

    def calc_read_write_counts_sequential(self):
        """
        Analytical Formula for Read/Write Counts Sequential
        #########################################################################################################

        WeightMemory and OutputMemory have full prefetches everytime so no need to worry about delta prefetches

        To determine the number of read counts at level n by level n-1:
            1) find full prefetch size of memory at level n-1
            2) examine loop tiling between level n and level n-1 memory
                a) find lowest dim that is in the memory's dependency set
                b) find product from lowest dim to second highest dim of topmost loop tile
            3) multiply product of dims with full prefetch size
        The result is the number of read counts for the first iteration of the entire convolution
        To find total read counts, multiply by range of highest dim


        InputMemory have both full and delta prefetches, so we need to analytically determine when a certain 
        prefetch scheme is used. When the lowest dim of an InputMemory is "input", then there will always be
        a delta prefetch when changing the val of a single dim ("weight" or "output") at a time. When changing
        the val of the next dimension, there is the potential for crossover (delta prefetch) and if not, we do 
        a full prefetch. When the lowest dim on an InputMemory is "channel", then there will never be a delta
        prefetch and so will always have full prefetches.

        To determine the number of read counts at level n by level n-1: 
            1) find full prefetch size of memory at level n-1
                a) determine what the lowest dim of memory at level n-1 is
                b) determine number of channels and spatial size of input space
            2) exmaine loop tiling between level n and n-1 memory
            3) if lowest dim == "channel"
                a) find "channel" dim and take product of all dims to second highest dim of topmost loop tile
                b) multiply product of dims with full prefetch size
            4) if lowest dim == "input"
                a) we know that the lowest dim in the loop tiling is either "weight" or "output"
                b) determine the associated spatial size of the lowest dim (i.e. take product of range of all same dims in lower loop tilings)
                    i) this is the delta size for each step in the lowest dim
                c) # of delta prefetches = lowest dim range - 1 (why -1: because first prefetch is always full prefetch)
                d) Case 1: if second lowest dim is either "weight" or "output"
                    i) determine the delta when crossing over from lowest dim to second lowest dim
                        a) delta = (second lowest dim spatial size) - (lowest dim spatial size * (lowest dim range - 1))
                        b) if delta is less than input spatial size of level n-1 memory, then can do delta prefetch
                            i) read counts:
                               (
                                full prefetch + 
                                (channel spatial size) * (lowest dim range - 1) * (lowest dim spatial size) * (second lowest dim range - 1) +
                                (channel spatial size) * ((second lowest dim spatial size) - (lowest dim spatial size * (lowest dim range - 1))) * (second lowest dim range - 1)
                               ) * 
                               product of all dimensions from second dim in loop tiling to second highest dim
                        c) else, need to do full prefetch
                            i) read counts:
                               (
                                full prefetch + 
                                (channel spatial size * (lowest dim range - 1) * lowest dim spatial size)
                               ) * 
                               product of all dimensions from third (second lowest) dim in loop tiling to second highest dim
                e) Case 2: if second lowest dim is "channel": need to do full prefetch when changing val of channel dim
                    i) read counts:
                       (
                        full prefetch + 
                        (channel spatial size) * (lowest dim range - 1) * (lowest dim spatial size)
                       ) * 
                       product of all dimensions from third (second lowest) dim in loop tiling to second highest dim

        If we are using write backs, then:
        # of read counts to level n from level n+/-1 = # of write counts to level n-1 from level n+/-1

        Total read counts at level n = # of read counts from level n-1 (prefetch) + # of read counts from level n+1 (writeback)
        """

        def calc_weight_read_count(level, loop_tiling_lst):
            # lowest memory is always written to
            if level == len(loop_tiling_lst) - 1: return 0

            # determine full prefetch size of level-1 memory
            channel_op_space_size, filter_op_space_size, weight_op_space_size = 1, 1, 1
            for loop_tile in loop_tiling_lst[level+1:]:
                for dim_dict in loop_tile:
                    for dim_name, dim_range in dim_dict.items():
                        if dim_name == "channel": channel_op_space_size *= dim_range
                        elif dim_name == "filter": filter_op_space_size *= dim_range
                        elif dim_name == "weight": weight_op_space_size *= dim_range
            full_prefetch_size = channel_op_space_size * filter_op_space_size * weight_op_space_size

            my_loop_tile = loop_tiling_lst[level]

            upper_dim_product = 1
            for loop_tile in loop_tiling_lst[:level]:
                for dim_dict in loop_tile:
                    dim_name, dim_range = list(dim_dict.items())[0]
                    upper_dim_product *= dim_range

            # determine the lowest dim
            lowest_dim = None
            for dim_dict in reversed(my_loop_tile):
                dim_name = list(dim_dict.keys())[0]
                if dim_name == "channel": 
                    lowest_dim = "channel"
                    break
                elif dim_name == "filter": 
                    lowest_dim = "filter"
                    break
                elif dim_name == "weight": 
                    lowest_dim = "weight"
                    break

            for dim_dict in my_loop_tile:
                dim_name, dim_range = list(dim_dict.items())[0]
                upper_dim_product *= dim_range
                if dim_name == lowest_dim: break

            read_count = upper_dim_product * full_prefetch_size
            return read_count


        def calc_output_read_count(level, loop_tiling_lst):
            # lowest memory is always written to
            if level == len(loop_tiling_lst) - 1: return 0

            # determine full prefetch size of level-1 memory
            filter_op_space_size, output_op_space_size = 1, 1
            for loop_tile in loop_tiling_lst[level+1:]:
                for dim_dict in loop_tile:
                    for dim_name, dim_range in dim_dict.items():
                        if dim_name == "filter": filter_op_space_size *= dim_range
                        elif dim_name == "output": output_op_space_size *= dim_range
            full_prefetch_size = filter_op_space_size * output_op_space_size

            my_loop_tile = loop_tiling_lst[level]

            upper_dim_product = 1
            for loop_tile in loop_tiling_lst[:level]:
                for dim_dict in loop_tile:
                    dim_name, dim_range = list(dim_dict.items())[0]
                    upper_dim_product *= dim_range

            # determine the lowest dim
            lowest_dim = None
            for dim_dict in reversed(my_loop_tile):
                dim_name = list(dim_dict.keys())[0]
                if dim_name == "filter": 
                    lowest_dim = "filter"
                    break
                elif dim_name == "output": 
                    lowest_dim = "output"
                    break

            for dim_dict in my_loop_tile:
                dim_name, dim_range = list(dim_dict.items())[0]
                upper_dim_product *= dim_range
                if dim_name == lowest_dim: break

            read_count = upper_dim_product * full_prefetch_size
            return read_count


        def calc_input_read_count(level, loop_tiling_lst):
            # lowest memory is always written to
            if level == len(loop_tiling_lst) - 1: return 0

            # determine full prefetch size of level-1 memory
            channel_op_space_size, weight_op_space_size, output_op_space_size = 1, 1, 1
            for loop_tile in loop_tiling_lst[level+1:]:
                for dim_dict in loop_tile:
                    for dim_name, dim_range in dim_dict.items():
                        if dim_name == "channel": channel_op_space_size *= dim_range
                        elif dim_name == "weight": weight_op_space_size *= dim_range
                        elif dim_name == "output": output_op_space_size *= dim_range
            input_op_space_size = weight_op_space_size + output_op_space_size - 1
            full_prefetch_size = channel_op_space_size * input_op_space_size

            my_loop_tile = loop_tiling_lst[level]

            upper_dim_product = 1
            for loop_tile in loop_tiling_lst[:level]:
                for dim_dict in loop_tile:
                    dim_name, dim_range = list(dim_dict.items())[0]
                    upper_dim_product *= dim_range

            # determine the lowest dim
            lowest_dim = None
            for dim_dict in reversed(my_loop_tile):
                dim_name = list(dim_dict.keys())[0]
                if dim_name == "channel": 
                    lowest_dim = "channel"
                    break
                elif dim_name == "weight" or dim_name == "output": 
                    lowest_dim = "input"
                    break

            # Case 1: lowest dim is channel
            if lowest_dim == "channel":

                for dim_dict in my_loop_tile:
                    dim_name, dim_range = list(dim_dict.items())[0]
                    upper_dim_product *= dim_range
                    if dim_name == "channel": break

                read_count = upper_dim_product * full_prefetch_size
                return read_count

            # Case 2: lowest dim is input
            else:
                is_consecutive_input = False
                lowest_input_dim_name = None
                lowest_input_dim_range = None
                second_lowest_dim_name = None
                second_lowest_input_dim_range = None

                # determine if output and weight dims are consecutive
                for i, dim_dict in enumerate(reversed(my_loop_tile)):
                    dim_name, dim_range = list(dim_dict.items())[0]
                    if dim_name == "weight" or dim_name == "output":
                        lowest_input_dim = dim_name
                        lowest_input_dim_range = dim_range

                        # get immediately prev dim
                        prev_dim_idx = len(my_loop_tile)-i-2
                        prev_dim_dict = my_loop_tile[prev_dim_idx]
                        prev_dim_name, prev_dim_range = list(prev_dim_dict.items())[0]
                        second_lowest_dim_name = prev_dim_name
                        second_lowest_input_dim_range = prev_dim_range
                        if prev_dim_name == "weight" or prev_dim_name == "output":
                            is_consecutive_input = True
                            break
                        else:
                            is_consecutive_input = False
                            break 

                # determine delta prefetches for lowest input dim 
                if lowest_input_dim == "weight":
                    delta_prefetch_size = weight_op_space_size * channel_op_space_size * (lowest_input_dim_range-1) * (second_lowest_input_dim_range)

                    # if is consecutive input, then need to determine dim crossover delta
                    if is_consecutive_input:
                        crossover_delta = output_op_space_size - (weight_op_space_size * (lowest_input_dim_range-1))

                    # channel was prev dim, so force a full prefetch
                    else:
                        crossover_delta = input_op_space_size
                else:
                    delta_prefetch_size = output_op_space_size * channel_op_space_size * (lowest_input_dim_range-1) * (second_lowest_input_dim_range)

                    # if is consecutive input, then need to determine dim crossover delta
                    if is_consecutive_input:
                        crossover_delta = weight_op_space_size - (output_op_space_size * (lowest_input_dim_range-1))

                    # channel was prev dim, so force a full prefetch
                    else:
                        crossover_delta = input_op_space_size

                # Case 2a: valid crossover delta
                if abs(crossover_delta) < input_op_space_size:
                    crossover_delta_prefetch_size = abs(crossover_delta) * channel_op_space_size * (second_lowest_input_dim_range-1)

                # Case 2b: no valid crossover delta
                else: 
                    crossover_delta_prefetch_size = full_prefetch_size * (second_lowest_input_dim_range-1)

                total_prefetch_size = full_prefetch_size + delta_prefetch_size + crossover_delta_prefetch_size

                # get product of all higher dims
                for dim_dict in my_loop_tile:
                    dim_name, dim_range = list(dim_dict.items())[0]
                    if dim_name == second_lowest_dim_name: break
                    upper_dim_product *= dim_range

                read_count = upper_dim_product * total_prefetch_size
                return read_count


        # create read/write counts dict
        input_read_write_counts_lst = []
        weight_read_write_counts_lst = []
        output_read_write_counts_lst = []

        def calc_read_count_wrapper(read_write_counts_lst, loop_tiling_lst, calc_read_count_func):
            for level in range(len(loop_tiling_lst)):
                read_write_count_dict = {"read count": 0, "write count": 0}

                # get forward read counts (read counts when level n memory is read by level n-1 memory)
                read_count = calc_read_count_func(level, loop_tiling_lst)
                read_write_count_dict["read count"] += read_count

                # get backward write counts (write counts when level n memory is written by level n+1 memory)
                if level != 0:
                    parent_read_write_count_dict = read_write_counts_lst[-1]
                    read_write_count_dict["write count"] += parent_read_write_count_dict["read count"]

                read_write_counts_lst.append(read_write_count_dict)

        calc_read_count_wrapper(input_read_write_counts_lst, self.loop_tiling_lst, calc_input_read_count)
        calc_read_count_wrapper(weight_read_write_counts_lst, self.loop_tiling_lst, calc_weight_read_count)
        calc_read_count_wrapper(output_read_write_counts_lst, self.loop_tiling_lst, calc_output_read_count)

        def calc_write_count_wrapper(read_write_counts_lst, write_backs, memory_type):
            # if write back is enabled, then:
            if memory_type in write_backs:
                for level in range(len(read_write_counts_lst)):
                    curr_read_write_count_dict = read_write_counts_lst[level]
                    # if not lowest memory, then can evaluate write backs
                    if level != len(read_write_counts_lst)-1:
                        child_read_write_count_dict = read_write_counts_lst[level+1]
                        curr_read_write_count_dict["write count"] += child_read_write_count_dict["write count"]
                        child_read_write_count_dict["read count"] += child_read_write_count_dict["write count"]

        calc_write_count_wrapper(input_read_write_counts_lst, self.write_backs, "input")
        calc_write_count_wrapper(weight_read_write_counts_lst, self.write_backs, "weight")
        calc_write_count_wrapper(output_read_write_counts_lst, self.write_backs, "output")

        print("Input")
        print(input_read_write_counts_lst)
        print()

        print("Weight")
        print(weight_read_write_counts_lst)
        print()

        print("Output")
        print(output_read_write_counts_lst)
        print()




