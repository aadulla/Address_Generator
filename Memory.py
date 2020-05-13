import numpy as np
import copy
import math
import random

from MiniMemory import MiniMemory

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
"Memory" represents the data stored and operated on at a level in the loop hierarchy. A memory 
instance consists of a list of mini memory instances corresponding to the subregions present
in the memory's op_space
"""
class Memory:
    
    """
    "__init__" initializes the memory
    params:
        memory_size:        physical size of the memory

        op_space_size:      size of the op_space the memory is responsible for

        level:              level in the hierarchy at which the memory is

        trace_queue:        queue that creats a sequential trace of all memory operations across 
                            all memories in the memory hierarchies

        cost:               cost associated with accessing this memory

        should_write_back:  indicates whether this memory should write back to the parent
                            memory on evictions

        write_back_policy:  determines what policy ("overwrite", "increment") to employ when
                            writing back


    """
    def __init__(self,
                 memory_type, 
                 memory_size, 
                 op_space_size, 
                 level,
                 trace_queue, 
                 cost, 
                 should_write_back=False, 
                 write_back_policy=None):
        
        self.memory_type = memory_type
        self.memory_size = memory_size
        self.op_space_size = op_space_size
        self.level = level
        self.trace_queue = trace_queue
        self.cost = cost
        self.should_write_back = should_write_back
        self.write_back_policy = write_back_policy
        
        # initialize an empty memory array
        self.memory_array = [None]*self.memory_size
        self.curr_op_space_start_ptr = -1
        self.curr_op_space_end_ptr = 0
        
        self.write_count = 0
        self.read_count = 0

        # buffers for communication between parent and child. these are initialized when the 
        # memory hierarchy is constructed
        self.read_backward_buffer = None
        self.read_forward_buffer = None
        self.write_backward_buffer = None
        self.write_forward_buffer = None

        self.prev_loop_counters = None
    
        
    """
    "clear_memory" clears the memory
    params: None
    """
    def clear_memory(self):
        for i in range(len(self.memory_array)):
            self.memory_array[i] = None
        self.curr_op_space_start_ptr = -1
            
    """
    "initialize" lays out data into the memory_array in row-major order
    params:
        data: the data to initialize the memory_array with
    """
    def initialize(self, data):

        def flatten(lst):
            if lst == []:
                return lst
            if isinstance(lst[0], list):
                return flatten(lst[0]) + flatten(lst[1:])
            return lst[:1] + flatten(lst[1:])
        
        flat_data = flatten(copy.copy(data))
        
        # place flattened data into memory_array
        for val in flat_data:
            self.memory_array[self.curr_op_space_end_ptr] = val
            self.curr_op_space_end_ptr += 1
        # set curr_op_space_ptr to 0 to indicate the memory_array is no longer empty
        self.curr_op_space_start_ptr = 0

    def get_lowest_dim(self):
        return list(self.dim_map[-1].items())[0][0]

            
    """
    "prefetch" determines how the memory should prefetch data from the parent memory as well as
    what data to prefetch. There are 2 kinds of prefetches: positive delta prefetch and negative
    delta prefetch.

    A positive delta prefetch is when the starting index of the contiguous block of data that is to 
    be prefetched is greater than the starting index of the contiguous block of data currently in
    the memory (i.e. the data to be prefetched is physically located at a later index in the parent
    memory).

    A negative delta prefetch is when the starting index of the contiguous block of data that is to 
    be prefetched is less than the starting index of the contiguous block of data currently in
    the memory (i.e. the data to be prefetched is physically located at an earlier index in the parent
    memory). 

    For instances of WeightMemory and OutputMemory, there will always be positive delta prefetches
    where there is no possible reuse of data between successive prefetches (i.e. delta == op_space_size)

    For instances of InputMemory, there will be a combination of positive and negative delta prefetches
    where there is a) possible reuse of data between successive prefetches (i.e |delta| < op_space_size),
    and b) there is no possible reuse of data between successive prefetches (i.e |delta| >= op_space_size)

    params:
        delta:              the difference between the starting indices in the parent memory of this
                            prefetch and the previous prefetch

        loop_counters:      dict mapping loop counter dimension names to their current values, only
                            has the relevant loop counters (i.e. loop counters one level higher)

        prev_loop_counters: dict mapping loop counter dimension names to the values at which the memory
                            last prefetched, only has the relevant loop counters 
                            (i.e. loop counters one level higher)

        loop_order_lst:     2D list where each 1D list corresponds to a loop block. The first element
                            in the 2D list is a 1D list corresponding to the loop block at the 
                            outermost level; the last elment in the 2D list is a 1D list 
                            corresponding to the loop block at the innermost level. The fist element 
                            in the 1D list corresponds to the outermost dimension in that loop block; 
                            the last element in the 1D list corresponds to the innermost dimension 
                            in that loop block
    """
    def prefetch(self, delta, loop_counters, loop_order_lst):
        # get the loop counter dimension that addresses the individual indicies inside a mini memory
        my_lowest_dim, my_lowest_op_space_size = list(self.dim_map[-1].items())[0]
        
        # get the lowest loop counter dimension in the loop block directly above the memory
        parent_lowest_dim = loop_order_lst[-1]
        
        # "weight" and "output" dimensions can be grouped in the "input" dimension
        if isinstance(self, InputMemory):
            if loop_order_lst[-1] == "weight" or loop_order_lst[-1] == "output":
                if my_lowest_dim == "channel":
                    parent_lowest_dim = "input"
                elif loop_order_lst[-2] == "channel":
                    if (self.prev_loop_counters is None) or (self.prev_loop_counters["channel"] == loop_counters["channel"]):
                        parent_lowest_dim = "input"
                    else:
                        parent_lowest_dim = "channel"
                else:
                    parent_lowest_dim = "input"
            else:
                parent_lowest_dim = "channel"
        
        # no data can be reused so we need to write back all the data currently in the memory and
        # then prefetch everything from the parent memory--this handles the case of prefetches
        # for WeightMemory and OutputMemory, fresh prefetches/dimension changes for InputMemory
        # if ((delta is None) or ((delta == 0) and (self.prev_loop_counters["channel"] != loop_counters["channel"]))):
        if delta is None:
            # check if memory array is not empty
            if self.curr_op_space_start_ptr != -1: 
                self.write_back_op_space()

            # prefetch everything new from parent
            num_to_prefetch = self.op_space_size // len(self.mini_memory_lst)
            self.positive_delta_prefetch(num_to_prefetch, None, loop_counters, True)
            self.curr_op_space_start_ptr = 0
            self.curr_op_space_end_ptr = 0
            self.prev_loop_counters = copy.deepcopy(loop_counters)

            return 1

        # no change in op_space
        elif delta == 0:
            return 0

        # if the dimensions are not the same, then we are moving to a completely different data space 
        # so we so need to prefetch everything new
        elif my_lowest_dim != parent_lowest_dim:
            # check if memory array is not empty
            if self.curr_op_space_start_ptr != -1:
                self.write_back_op_space()

            # prefetch everything new from parent
            num_to_prefetch = self.op_space_size // len(self.mini_memory_lst)
            self.positive_delta_prefetch(num_to_prefetch, None, loop_counters, True)
            self.curr_op_space_start_ptr = 0
            self.curr_op_space_end_ptr = 0
            self.prev_loop_counters = copy.deepcopy(loop_counters)

            return 1
        
        # since the dimensions are the same, we are in the same subspace so can utilize delta prefetch
        elif my_lowest_dim == parent_lowest_dim:
            # only input memory can have delta prefetches
            assert isinstance(self, InputMemory)

            # the delta is too big (i.e. the sub-subspaces are disconnected) so we need to prefectch 
            # everything new
            if abs(delta) >= my_lowest_op_space_size:
                # check if memory array is not empty
                if self.curr_op_space_start_ptr != -1:
                    self.write_back_op_space()

                # prefetch everything new from parent
                num_to_prefetch = self.op_space_size // len(self.mini_memory_lst)
                self.positive_delta_prefetch(num_to_prefetch, None, loop_counters, True)
                self.curr_op_space_start_ptr = 0
                self.curr_op_space_end_ptr = 0
                self.prev_loop_counters = copy.deepcopy(loop_counters)

                return 1
            
            # negative delta prefetch (i.e. we move our sub-subspace backward in the subspace)
            elif delta < 0:
                num_to_prefetch = abs(delta)
                self.negative_delta_prefetch(num_to_prefetch, self.prev_loop_counters, loop_counters, False)
                   
                # need to adjust ptrs in actual memory object
                self.curr_op_space_start_ptr = (self.curr_op_space_start_ptr + delta) % self.memory_size
                self.curr_op_space_end_ptr = (self.curr_op_space_end_ptr + delta) % self.memory_size
                # self.prev_loop_counter = copy.deepcopy(loop_counters)

                return 0
                
            # positive delta prefetch (i.e we move our sub-subspace forward in the subspace)
            elif delta > 0 and delta < self.mini_memory_lst[0].op_space_size:
                num_to_prefetch = abs(delta)
                self.positive_delta_prefetch(num_to_prefetch, self.prev_loop_counters, loop_counters, False)
                
                # need to adjust ptrs in actual memory object
                self.curr_op_space_start_ptr = (self.curr_op_space_start_ptr + delta) % self.memory_size
                self.curr_op_space_end_ptr = (self.curr_op_space_end_ptr + delta) % self.memory_size
                # self.prev_loop_counter = copy.deepcopy(loop_counters)

                return 0
                
            # should never come to this case
            else:
                print("Error, should never come into this case")
                assert False

    """
    "extract_idx_from_request" determines what index in the memory_array corresponds to the request
    params:
        request: dictionary mapping loop counter dimensions to their value
    """
    def extract_idx_from_request(self, request):
        # transform the request to have the diemsions be in the same order of how the data in the 
        # memory is laid out
        my_idxs = self.transform_request(request)

        # if memory is indexed by 2 dimensions (i.e. output and input memories)
        if len(my_idxs) == 2:
            mini_memory_idx = my_idxs[0]
            data_idx = self.mini_memory_lst[mini_memory_idx].curr_op_space_start_ptr + my_idxs[1]

        # if memory is indexed by 3 dimensions (i.e. weight memories)
        elif len(my_idxs) == 3:
            _, dim_1_size = list(self.dim_map[1].items())[0]
            mini_memory_idx = int(my_idxs[0]*dim_1_size + my_idxs[1])
            data_idx = self.mini_memory_lst[mini_memory_idx].curr_op_space_start_ptr + my_idxs[2]
            
        return data_idx % self.memory_size

    """
    "write_backward" writes the data located at my_memory_idx back to the parent memory at the
    location specified in write_request. This is for handling data that is sent from the child to 
    parent memory in a write back.
    params:
        my_memory_idx:  index of data in memory array to write back

        write_request:  dictionary containing information about what index in the parent memory array 
                        is being written in
    """
    def write_backward(self, my_memory_idx, write_request):
        # check if we should write back
        if self.should_write_back:
            # update trace queue
            command = {"level": self.level, "op": "write backward", "address": my_memory_idx}
            self.trace_queue.enque(command)

            data = self.memory_array[my_memory_idx]
            # put data into the buffer
            self.write_backward_buffer.enque(data)
            # send data to parent
            self.write_backward_buffer.send_data_child_to_parent(write_request)

            # data value is getting read out of memory
            self.read_count += 1

        
        # set the data at my_memory_idx to None to indicate it is empty/was written back
        self.memory_array[my_memory_idx] = None

    """
    "read_forward" reads the data from the read_forward_buffer into the index in the memory array
    specified by read_request. This is for handling data that is sent from the child to parent memory
    in a write back.
    params:
        read_request:   dictionary containing information about what index in the parent memory array 
                        the data should be read into
    """
    def read_forward(self, read_request):
        # convert the read_request into a physical index into the memory array
        my_memory_idx = self.extract_idx_from_request(read_request)  
        # get data from the buffer                          
        data = self.read_forward_buffer.deque()
        assert data is not None

        # read in data following write back policy
        if self.write_back_policy == "overwrite":
            self.memory_array[my_memory_idx] = data
        elif self.write_back_policy == "increment":
            self.memory_array[my_memory_idx] += data

        # update trace queue
        command = {"level": self.level, "op": "read forward", "address": my_memory_idx}
        self.trace_queue.enque(command)

        # data value is getting written to in memory
        self.write_count += 1
    
    """
    "read_backward" reads the data from the read_backward_buffer into the index in the memory array
    specified by my_memory_idx. This is for handling data that is sent from the parent to child 
    memory in a prefetch.
    params:
        my_memory_idx:  index in memory_array data should be read into

        read_request:   dictionary containing information about what index in the memory array data
                        is being read from in the parent
    """
    def read_backward(self, my_memory_idx, read_request):
        # issue read request to parent so parent knows what value to send down
        self.read_backward_buffer.send_data_parent_to_child(read_request)
        # get data from buffer
        data = self.read_backward_buffer.deque()
        # put data into memory array
        self.memory_array[my_memory_idx] = data

        # update trace queue
        command = {"level": self.level, "op": "read backward", "address": my_memory_idx}
        self.trace_queue.enque(command)
        
        # data value is getting written to in memory
        self.write_count += 1
        
    """
    "write_forward" writes the data located at the index specified in write_request. This is for
    handling data that is sent from parent to child memory in a prefetch.
    params:
        write_request:  dictionary containing information about what index in the memory array data
                        is being written from
    """
    def write_forward(self, write_request):
        # get physical index into memory array from write_request
        my_memory_idx = self.extract_idx_from_request(write_request) 
        data = self.memory_array[my_memory_idx]
        # put data into buffer
        self.write_forward_buffer.enque(data)

        # update trace queue
        command = {"level": self.level, "op": "write forward", "address": my_memory_idx}
        self.trace_queue.enque(command)
        
        # data value is getting read out of memory
        self.read_count += 1

    """
    "write_back_op_space" writes back all the data in the current memory to its parent memory
    param:
        prev_loop_counters: dictionary mapping the loop counters immediately above the memory (i.e. 
                            one level above) to the values at which the memory began a fresh prefetch
                            of data from the parent memory. The pointers in the memory and its mini
                            memories were initialized with reference to these loop counters
    """
    def write_back_op_space(self):
        my_memory_idxs, parent_memory_requests = [], []
        
        # iterate through each mini memory and get the indices (my_memory_idxs) of the data to 
        # write back and the requests into the parent memory (parent_memory_requests) that these
        # indices map to
        for i, mini_memory in enumerate(self.mini_memory_lst):
            tmp_memory_idxs, tmp_parent_memory_idxs = mini_memory.write_back_op_space()
            tmp_parent_memory_requests = self.convert_idxs_to_requests(i, tmp_parent_memory_idxs, self.prev_loop_counters)
            my_memory_idxs.extend(tmp_memory_idxs)
            parent_memory_requests.extend(tmp_parent_memory_requests)
            
        # write back data
        for my_memory_idx, parent_memory_request in zip(my_memory_idxs, parent_memory_requests):
            self.write_backward(my_memory_idx, parent_memory_request)
                
        self.clear_memory()
        self.reset_ptrs()
            
    """
    "load" returns data located at the index specified by request
    params:
        request: dictionary containing information about what index in the memory array data
                 is being written from
    """
    def load(self, request):
        if "input" in request.keys(): is_input = True
        else: is_input = False
        my_memory_idx = self.extract_idx_from_request(request)
        # self.write_count += 1

        # update trace queue
        command = {"level": self.level, "op": "load", "address": my_memory_idx}
        self.trace_queue.enque(command)

        return self.memory_array[my_memory_idx]
    
    """
    "store" stores val at the index specified by request
    params:
        val: data to be read in

        request: dictionary containing information about what index in the memory array data
                 is being written to
    """
    def store(self, request, val):
        my_memory_idx = self.extract_idx_from_request(request)
        # self.read_count += 1
        self.memory_array[my_memory_idx] = val

        # update trace queue
        command = {"level": self.level, "op": "store", "address": my_memory_idx}
        self.trace_queue.enque(command)

    """ 
    "reset_ptrs" resets the pointers of the memory
    params: None
    """
    def reset_ptrs(self):
        self.curr_op_space_start_ptr = -1
        self.curr_op_space_end_ptr = 0
        for mini_memory in self.mini_memory_lst:
            mini_memory.reset_ptrs()

    # barrier syncrhonization between different memory streams (input, weight, output)
    def barrier_sync(self):
        self.trace_queue.enque("BARRIER")

    # end trace
    def end_trace(self):
        self.trace_queue.enque("END")
    
    """ 
    "calc_cost" calculates the total cost of all the accesses to memory
    params: None
    """    
    def calc_cost(self):
        return (self.write_count + self.read_count)*self.cost
       
    """
    "print" prints information about the memory
    params: None
    """ 
    def print(self):
        print("Memory Size:", self.memory_size)
        print("Op Space Size:", self.op_space_size)
        print("Memory Array: ", end="")
        print(self.memory_array)
        
        print("Curr Start Pointer:", self.curr_op_space_start_ptr)
        print("End Pointer:", self.curr_op_space_end_ptr)
        
        print("Mini Memories:")
        for mini_memory in self.mini_memory_lst:
            mini_memory.print()
        
        print("Read Buffers:")
        print("\t Backward Buffer: ", end="")
        print(self.read_backward_buffer)
        print("\t Forward Buffer: ", end="")
        print(self.read_forward_buffer)
        
        print("Write Buffers:")
        print("\t Backward Buffer: ", end="")
        print(self.write_backward_buffer)
        print("\t Forward Buffer: ", end="")
        print(self.write_forward_buffer)
        
        print("Read Count:", self.read_count)
        print("Write Count:", self.write_count)
        
        print("Cost:", self.calc_cost())
    
    """
    "print_stats" prints information about the memory accesses
    params: None
    """ 
    def print_stats(self, pad=""):
        print(pad + "Read Count: " + str(self.read_count))
        print(pad +  "Write Count: " + str(self.write_count))
        print(pad + "Cost: " +  str(self.calc_cost()))

"""
"InputMemory" is a subclass of memory designed specifcally to interface with input data
"""
class InputMemory(Memory):

    """
    "__init__" initializes the memory
    params:
        memory_size:        physical size of the memory

        op_space_size:      size of the op_space the memory is responsible for

        upper_dim_map:      list of dictionaries mapping loop counters to the sizes of their op spaces
                            in the loop block directly above the memory

        cost:               cost associated with accessing this memory

        should_write_back:  indicates whether this memory should write back to the parent
                            memory on evictions

        write_back_policy:  determines what policy ("overwrite", "increment") to employ when
                            writing back
    """
    def __init__(self, 
                 memory_size, 
                 op_space_size,
                 level, 
                 trace_queue, 
                 upper_dim_map,
                 cost,
                 should_write_back=False, 
                 write_back_policy=None):
        
        super().__init__("input", memory_size, op_space_size, level, trace_queue, cost, should_write_back, write_back_policy)
        # the dimensions which the memory is addressed by
        self.dependency_set = {"channel", "weight", "output"}
        
        self.dim_map_dict = {}
        input_op_space_size = 0
        channel_op_space_size = 0
        weight_op_space = 0
        
        # is_input_last indicates whether the last/lowermost dimension in the loop block above
        # is either "weight" or "output". this is useful when performing prefetches to determine
        # how to transform the request
        self.is_input_last = False

        dim_count = 0
        # iterate through all the dimensions in the loop block above
        for dim in upper_dim_map:
            dim_name, dim_size = list(dim.items())[0]
            # check if dimension affects the memory
            if dim_name in self.dependency_set:
                dim_count += 1
                self.dim_map_dict.update(dim)
                if dim_name == "weight": 
                    input_op_space_size += dim_size
                    if dim_count == 3 : self.is_input_last = True
                if dim_name == "output": 
                    input_op_space_size += dim_size
                    if dim_count == 3 : self.is_input_last = True
                if dim_name == "channel": 
                    channel_op_space_size += dim_size

        # check if last dimension in loop block was either "weight" or "output" and format
        # the order in dim_map accordingly
        if (self.is_input_last):
            dim_0_size = channel_op_space_size
            dim_1_size = input_op_space_size - 1
            self.dim_map = [{"channel": channel_op_space_size}, {"input": input_op_space_size}]
        else:
            dim_0_size = input_op_space_size - 1
            dim_1_size = channel_op_space_size
            self.dim_map = [{"input": input_op_space_size}, {"channel": channel_op_space_size}]
            
        num_mini_memories = dim_0_size
        
        # evenly distribute the extra space in the memory array as evenly as posible between
        # the mini memories
        extra_space = memory_size - (dim_0_size * dim_1_size)
        even_padding = extra_space // num_mini_memories
        uneven_padding = extra_space % num_mini_memories
        
        offsets = [0]*num_mini_memories
        sizes   = [0]*num_mini_memories
        for i in range(num_mini_memories):
            if uneven_padding > 0:
                to_add = 1
                uneven_padding -= 1
            else:
                to_add = 0
            offsets[i] = offsets[i-1] + sizes[i-1]
            sizes[i] = dim_1_size + even_padding + to_add
            
        # initialize the mini memories in mini_memory_lst
        self.mini_memory_lst = []
        for i in range(num_mini_memories):
            self.mini_memory_lst.append(MiniMemory(offsets[i], dim_1_size, self.memory_size))

        self.mini_memory_op_space_size = dim_1_size
        self.num_mini_memories = num_mini_memories
            
    """
    "transform_request" converts the request into a list of tuple of indices that can be used to
    index into a specific mini memory and then an index inside that mini memory
    params:
        request:    dictionary containing information about the index in the memory that is requested
                    to either be read from or written to
    """
    def transform_request(self, request):
        # get the order of dimensions of how data is laid out in meomry
        my_dim_0_name, _ = list(self.dim_map[0].items())[0]
        my_dim_1_name, _ = list(self.dim_map[1].items())[0]
        
        # get indexes from the request
        my_idx_0 = request[my_dim_0_name]
        my_idx_1 = request[my_dim_1_name]
        
        return my_idx_0, my_idx_1
    
    """
    "convert_idx_to_requests" converts indexes into the parent memory relative to certain loop 
    counters into requests which the parent memory can use to extract the absolute indices in its
    memory array
    params:
        mini_memory_idx:    the index of the mini memory in mini_memory_lst

        parent_memory_idxs: list of relative indices into the parent memory

        prev_loop_counters: dictionary mapping loop counter dimensions to their values from which
                            parent_memory_idxs are reference to
    """
    def convert_idxs_to_requests(self, mini_memory_idx, parent_memory_idxs, prev_loop_counters):
        parent_memory_requests = []
        
        for parent_memory_idx in parent_memory_idxs:
            
            if (self.is_input_last):
                dim_0_name = "channel"
                dim_0_offset = prev_loop_counters["channel"] * self.dim_map_dict["channel"] + mini_memory_idx
                dim_1_name = "input"
                dim_1_offset = (prev_loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                prev_loop_counters["output"] * self.dim_map_dict["output"]) + parent_memory_idx
            else:
                dim_0_name = "input"
                dim_0_offset = (prev_loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                prev_loop_counters["output"] * self.dim_map_dict["output"]) + mini_memory_idx
                dim_1_name = "channel"
                dim_1_offset = prev_loop_counters["channel"] * self.dim_map_dict["channel"] + parent_memory_idx

            request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}
            parent_memory_requests.append(request)
        
        return parent_memory_requests
     
    """
    "negative_delta_prefetch" performs a prefetch in the backwards direction
    params:
        num_to_prefetch:        the number of elements to prefetch

        delta_loop_counters:    a dictionary mapping loop counters to their values at which the last
                                prefetch was done

        loop_counters:          a dictionary mapping loop counters to their current values

        is_static:              determines if the curr_start_ptr should be updated
    """       
    def negative_delta_prefetch(self, num_to_prefetch, delta_loop_counters, loop_counters, is_static):
        # create a reverse copy of the mini memory lst
        reversed_curr_mini_memory_lst = list(reversed(self.mini_memory_lst))
        reversed_prev_mini_memory_lst = reversed_curr_mini_memory_lst[1:]
        reversed_prev_mini_memory_lst.append(reversed_curr_mini_memory_lst[0])
        jump = num_to_prefetch
        
        # iterate through all the elements to prefetch
        for i in range(num_to_prefetch):
            # iterate through all the mini memories
            for j, (curr_mini_memory, prev_mini_memory) in enumerate(zip(reversed_curr_mini_memory_lst, reversed_prev_mini_memory_lst)):
                # evaluate if there is an intersection between curr mini memory and prev mini memory
                is_intersecting, my_memory_idx, parent_idx, mini_memory_offset = curr_mini_memory.is_intersecting_backward(prev_mini_memory)
                
                reversed_j = self.num_mini_memories - j - 1

                # check if there was in intersection
                if is_intersecting:
                    # create the write request
                    dim_0_name = "channel"
                    dim_0_offset = delta_loop_counters["channel"] * self.dim_map_dict["channel"] + \
                                  (reversed_j + mini_memory_offset) % self.num_mini_memories
                    dim_1_name = "input"
                    dim_1_offset = (delta_loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                    delta_loop_counters["output"] * self.dim_map_dict["output"]) + parent_idx
                    write_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}

                    # send data back to the parent
                    self.write_backward(my_memory_idx, write_request)

                # create the read request
                dim_0_name = "channel"
                dim_0_offset = loop_counters["channel"] * self.dim_map_dict["channel"] + \
                               reversed_j % self.num_mini_memories
                dim_1_name = "input"
                dim_1_offset = (loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                loop_counters["output"] * self.dim_map_dict["output"])  + jump - i - 1
                read_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}

                # read data from the parent
                self.read_backward(my_memory_idx, read_request)

                # adjust the pointers
                curr_mini_memory.decrement_ptrs(is_static)

    """
    "positive_delta_prefetch" performs a prefetch in the backwards direction
    params:
        num_to_prefetch:        the number of elements to prefetch

        delta_loop_counters:    a dictionary mapping loop counters to their values at which the last
                                prefetch was done

        loop_counters:          a dictionary mapping loop counters to their current values

        is_static:              determines if the curr_start_ptr should be updated
    """             
    def positive_delta_prefetch(self, num_to_prefetch, delta_loop_counters, loop_counters, is_static):
        # create a rotated right copy of the mini memory lst
        next_mini_memory_lst = self.mini_memory_lst[1:]
        next_mini_memory_lst.append(self.mini_memory_lst[0])
        jump = self.mini_memory_op_space_size - num_to_prefetch

        # iterate through all the elements to prefetch
        for i in range(num_to_prefetch):
            # iterate through all the mini memories
            for j, (curr_mini_memory, next_mini_memory) in enumerate(zip(self.mini_memory_lst, next_mini_memory_lst)):
                # evaluate if there is an intersection between curr mini memory and next mini memory
                is_intersecting, my_memory_idx, parent_idx, mini_memory_offset = curr_mini_memory.is_intersecting_forward(next_mini_memory)
                
                # check if there was in intersection
                if is_intersecting:
                    # create the write request
                    if (self.is_input_last):
                        dim_0_name = "channel"
                        dim_0_offset = delta_loop_counters["channel"] * self.dim_map_dict["channel"] + \
                                      (j + mini_memory_offset) % self.num_mini_memories
                        dim_1_name = "input"
                        dim_1_offset = (delta_loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                        delta_loop_counters["output"] * self.dim_map_dict["output"]) + parent_idx
                    else:
                        dim_0_name = "input"
                        dim_0_offset = (delta_loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                        delta_loop_counters["output"] * self.dim_map_dict["output"]) + \
                                       (j + mini_memory_offset) % self.num_mini_memories
                        dim_1_name = "channel"
                        dim_1_offset = delta_loop_counters["channel"] * self.dim_map_dict["channel"] + parent_idx

                    write_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}

                    # send data back to the parent
                    self.write_backward(my_memory_idx, write_request)
                                    
                # create the read request
                if (self.is_input_last):
                    dim_0_name = "channel"
                    dim_0_offset = loop_counters["channel"] * self.dim_map_dict["channel"] + j
                    dim_1_name = "input"
                    dim_1_offset = (loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                    loop_counters["output"] * self.dim_map_dict["output"]) + jump + i
                else:
                    dim_0_name = "input"
                    dim_0_offset = (loop_counters["weight"] * self.dim_map_dict["weight"] + \
                                    loop_counters["output"] * self.dim_map_dict["output"]) + j
                    dim_1_name = "channel"
                    dim_1_offset = loop_counters["channel"] * self.dim_map_dict["channel"] + jump + i

                read_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}

                # read data from the parent
                self.read_backward(my_memory_idx, read_request)
                
                # adjust the pointers
                curr_mini_memory.increment_ptrs(is_static)

"""
"WeightMemory" is a subclass of memory designed specifcally to interface with weight data
"""    
class WeightMemory(Memory):

    """
    "__init__" initializes the memory
    params:
        memory_size:        physical size of the memory

        op_space_size:      size of the op_space the memory is responsible for

        level:              level in the hierarchy at which the memory is

        trace_queue:        queue that creats a sequential trace of all memory operations across 
                            all memories in the memory hierarchies

        upper_dim_map:      list of dictionaries mapping loop counters to the sizes of their op spaces
                            in the loop block directly above the memory

        cost:               cost associated with accessing this memory

        should_write_back:  indicates whether this memory should write back to the parent
                            memory on evictions

        write_back_policy:  determines what policy ("overwrite", "increment") to employ when
                            writing back
    """
    def __init__(self, 
                 memory_size, 
                 op_space_size,
                 level, 
                 trace_queue, 
                 upper_dim_map,
                 cost,
                 should_write_back=False, 
                 write_back_policy=None):
        
        super().__init__("weight", memory_size, op_space_size, level, trace_queue, cost, should_write_back, write_back_policy)
        self.dependency_set = {"channel", "filter", "weight"}
        
        self.dim_map = []
        for dim in upper_dim_map:
            dim_name, dim_size = list(dim.items())[0]
            if dim_name in self.dependency_set:
                self.dim_map.append(dim)

        _, dim_0_size = list(self.dim_map[0].items())[0]
        _, dim_1_size = list(self.dim_map[1].items())[0]
        _, dim_2_size = list(self.dim_map[2].items())[0]
        
        num_mini_memories = dim_0_size * dim_1_size
        
        extra_space = memory_size - (dim_0_size * dim_1_size * dim_2_size)
        even_padding = extra_space // num_mini_memories
        uneven_padding = extra_space % num_mini_memories
    
        offsets = [0]*num_mini_memories
        sizes   = [0]*num_mini_memories
        for i in range(num_mini_memories):
            if uneven_padding > 0:
                to_add = 1
                uneven_padding -= 1
            else:
                to_add = 0
            offsets[i] = offsets[i-1] + sizes[i-1]
            sizes[i] = dim_2_size + even_padding + to_add
            
        self.mini_memory_lst = []
        for i in range(num_mini_memories):
            self.mini_memory_lst.append(MiniMemory(offsets[i], dim_2_size, self.memory_size))
        
        self.mini_memory_op_space_size = dim_2_size
        self.num_mini_memories = num_mini_memories
                                    
    def transform_request(self, request):
        my_dim_0_name, _ = list(self.dim_map[0].items())[0]
        my_dim_1_name, _ = list(self.dim_map[1].items())[0]
        my_dim_2_name, _ = list(self.dim_map[2].items())[0]
        
        my_idx_0 = request[my_dim_0_name]
        my_idx_1 = request[my_dim_1_name]
        my_idx_2 = request[my_dim_2_name]
        
        return my_idx_0, my_idx_1, my_idx_2
    
    def convert_idxs_to_requests(self, mini_memory_idx, parent_memory_idxs, prev_loop_counters):
        parent_memory_requests = []
        dim_0_name, dim_0_size = list(self.dim_map[0].items())[0]
        dim_1_name, dim_1_size = list(self.dim_map[1].items())[0]
        dim_2_name, dim_2_size = list(self.dim_map[2].items())[0]
        
        for parent_memory_idx in parent_memory_idxs:
            dim_0_offset = prev_loop_counters[dim_0_name] * dim_0_size + mini_memory_idx//dim_1_size
            dim_1_offset = prev_loop_counters[dim_1_name] * dim_1_size + mini_memory_idx%dim_1_size
            dim_2_offset = prev_loop_counters[dim_2_name] * dim_2_size + parent_memory_idx
            request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset, dim_2_name: dim_2_offset}
            
            parent_memory_requests.append(request)
        
        return parent_memory_requests
                
    def positive_delta_prefetch(self, num_to_prefetch, delta_loop_counters, loop_counters, is_static):
        next_mini_memory_lst = self.mini_memory_lst[1:]
        next_mini_memory_lst.append(self.mini_memory_lst[0])
        
        for i in range(num_to_prefetch):
            for j, (curr_mini_memory, next_mini_memory) in enumerate(zip(self.mini_memory_lst, next_mini_memory_lst)):
                is_intersecting, my_memory_idx, parent_idx, _ = curr_mini_memory.is_intersecting_forward(next_mini_memory)

                dim_0_name, dim_0_size = list(self.dim_map[0].items())[0]
                dim_1_name, dim_1_size = list(self.dim_map[1].items())[0]
                dim_2_name, dim_2_size = list(self.dim_map[2].items())[0]
                
                if is_intersecting:
                    dim_0_offset = loop_counters[dim_0_name] * dim_0_size + j//dim_1_size
                    dim_1_offset = loop_counters[dim_1_name] * dim_1_size + j%dim_1_size
                    dim_2_offset = loop_counters[dim_2_name] * dim_2_size + parent_idx
                    write_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset, dim_2_name: dim_2_offset}
                    self.write_backward(my_memory_idx, write_request)
                
                dim_0_offset = loop_counters[dim_0_name] * dim_0_size + j//dim_1_size
                dim_1_offset = loop_counters[dim_1_name] * dim_1_size + j%dim_1_size
                dim_2_offset = loop_counters[dim_2_name] * dim_2_size + curr_mini_memory.parent_map + (curr_mini_memory.curr_op_space_end_ptr - curr_mini_memory.curr_op_space_start_ptr) % self.memory_size
                read_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset, dim_2_name: dim_2_offset}
                self.read_backward(my_memory_idx, read_request)

                curr_mini_memory.increment_ptrs(is_static)
  
"""
"OutputMemory" is a subclass of memory designed specifcally to interface with output data
"""          
class OutputMemory(Memory):

    """
    "__init__" initializes the memory
    params:
        memory_size:        physical size of the memory

        op_space_size:      size of the op_space the memory is responsible for

        level:              level in the hierarchy at which the memory is

        trace_queue:        queue that creats a sequential trace of all memory operations across 
                            all memories in the memory hierarchies

        upper_dim_map:      list of dictionaries mapping loop counters to the sizes of their op spaces
                            in the loop block directly above the memory

        cost:               cost associated with accessing this memory

        should_write_back:  indicates whether this memory should write back to the parent
                            memory on evictions

        write_back_policy:  determines what policy ("overwrite", "increment") to employ when
                            writing back
    """
    def __init__(self, 
                 memory_size, 
                 op_space_size,
                 level, 
                 trace_queue, 
                 upper_dim_map,
                 cost,
                 should_write_back=False, 
                 write_back_policy=None):
        
        super().__init__("output", memory_size, op_space_size, level, trace_queue, cost, should_write_back, write_back_policy)
        self.dependency_set = {"filter", "output"}

        # only applicable for L0 memory
        self.prev_memory_array = [None]*memory_size
        
        self.dim_map = []
        for dim in upper_dim_map:
            dim_name, dim_size = list(dim.items())[0]
            if dim_name in self.dependency_set:
                self.dim_map.append(dim)

        _, dim_0_size = list(self.dim_map[0].items())[0]
        _, dim_1_size = list(self.dim_map[1].items())[0]
        
        num_mini_memories = dim_0_size
        
        extra_space = memory_size - (dim_0_size * dim_1_size)
        even_padding = extra_space // num_mini_memories
        uneven_padding = extra_space % num_mini_memories
        
        offsets = [0]*num_mini_memories
        sizes   = [0]*num_mini_memories
        for i in range(num_mini_memories):
            if uneven_padding > 0:
                to_add = 1
                uneven_padding -= 1
            else:
                to_add = 0
            offsets[i] = offsets[i-1] + sizes[i-1]
            sizes[i] = dim_1_size + even_padding + to_add
        
            
        self.mini_memory_lst = []
        for i in range(num_mini_memories):
            self.mini_memory_lst.append(MiniMemory(offsets[i], dim_1_size, self.memory_size))
            
        self.mini_memory_op_space_size = dim_1_size
        self.num_mini_memories = num_mini_memories
                                    
    def transform_request(self, request):
        my_dim_0_name, _ = list(self.dim_map[0].items())[0]
        my_dim_1_name, _ = list(self.dim_map[1].items())[0]
        
        my_idx_0 = request[my_dim_0_name]
        my_idx_1 = request[my_dim_1_name]
        
        return my_idx_0, my_idx_1
    
    def convert_idxs_to_requests(self, mini_memory_idx, parent_memory_idxs, prev_loop_counters):
        parent_memory_requests = []
        dim_0_name, dim_0_size = list(self.dim_map[0].items())[0]
        dim_1_name, dim_1_size = list(self.dim_map[1].items())[0]
        
        for parent_memory_idx in parent_memory_idxs:
            
            dim_0_offset = prev_loop_counters[dim_0_name] * dim_0_size + mini_memory_idx
            dim_1_offset = prev_loop_counters[dim_1_name] * dim_1_size + parent_memory_idx
            request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}
            
            parent_memory_requests.append(request)
        
        return parent_memory_requests
                
    def positive_delta_prefetch(self, num_to_prefetch, delta_loop_counters, loop_counters, is_static):
        next_mini_memory_lst = self.mini_memory_lst[1:]
        next_mini_memory_lst.append(self.mini_memory_lst[0])
        
        for i in range(num_to_prefetch):
            for j, (curr_mini_memory, next_mini_memory) in enumerate(zip(self.mini_memory_lst, next_mini_memory_lst)):
                is_intersecting, my_memory_idx, parent_idx, _ = curr_mini_memory.is_intersecting_forward(next_mini_memory)

                dim_0_name, dim_0_size = list(self.dim_map[0].items())[0]
                dim_1_name, dim_1_size = list(self.dim_map[1].items())[0]
                
                if is_intersecting:
                    dim_0_offset = loop_counters[dim_0_name] * dim_0_size + j
                    dim_1_offset = loop_counters[dim_1_name] * dim_1_size + parent_idx
                    write_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}
                    self.write_backward(my_memory_idx, write_request)
                
                dim_0_offset = loop_counters[dim_0_name] * dim_0_size + j
                dim_1_offset = loop_counters[dim_1_name] * dim_1_size + curr_mini_memory.parent_map + (curr_mini_memory.curr_op_space_end_ptr - curr_mini_memory.curr_op_space_start_ptr) % self.memory_size
                read_request = {dim_0_name: dim_0_offset, dim_1_name: dim_1_offset}
                self.read_backward(my_memory_idx, read_request)
                
                curr_mini_memory.increment_ptrs(is_static)

        self.prev_memory_array = copy.deepcopy(self.memory_array)

    def write_backward(self, my_memory_idx, write_request):
        # if L0 memory, then write back delta
        if self.level == 0:
            new_val = self.memory_array[my_memory_idx]
            old_val = self.prev_memory_array[my_memory_idx]
            delta_val = new_val - old_val
            self.memory_array[my_memory_idx] = delta_val

        super().write_backward(my_memory_idx, write_request)
