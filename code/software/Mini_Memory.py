import numpy as np
import copy
import math
import random

from .Data_Block import Data_Block

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
"Mini_Memory" corresponds to a contiguous subregion in the op_space of the current memory
"""
class Mini_Memory:

    """
    "__init__" initializes the Mini_Memory
    pararms:
        offset:         the offset in the actual memory where the mini memory starts

        op_space_size:  the op_space_size of the subregion encapsulated by the mini memory

        memory_size:    the physical size of the actual memory
    """
    def __init__(self, offset, op_space_size, memory_size):
        self.offset = offset
        self.op_space_size = op_space_size
        self.memory_size = memory_size
        
        # curr_op_space_start_ptr is a pointer to where the mini memory starts in the actual memory
        self.curr_op_space_start_ptr = offset
        # curr_op_space_end_ptr is a pointer to where the mini memory end in the actual memory
        self.curr_op_space_end_ptr = offset
        
        # pointer attached to curr_op_space_start_ptr that maps where curr_op_space_start_ptr is 
        # in the current memory to where it is in the parent memory
        self.parent_map = 0
        
        # datablock for old data that was indexed before curr_op_space_start_ptr
        self.backward_old_data_block = Data_Block(self.curr_op_space_start_ptr, self.memory_size)
        # datablock for old data that was indexed after curr_op_space_end_ptr
        self.forward_old_data_block = Data_Block(self.curr_op_space_end_ptr, self.memory_size)

    """
    "reset" resets the mini memory
    params: None
    """
    def reset_ptrs(self):
        self.curr_op_space_start_ptr = self.offset
        self.curr_op_space_end_ptr = self.offset
        self.parent_map = 0
        
        self.backward_old_data_block.reset()
        self.forward_old_data_block.reset()

    """
    "write_back_op_space" writes back all the old and current data enpasulated by the mini memory.
    It returns 2 lists: the first references the indices of the data in the actual memory that are 
    to be written back from, the second references the indices of data in the parent memory that are 
    """
    def write_back_op_space(self):
        my_memory_idxs = []
        parent_memory_idxs = []

        # get old data to write backward from backward_old_data_block
        for i in range(self.backward_old_data_block.size):
            my_memory_idxs.append(self.backward_old_data_block.start)
            parent_memory_idxs.append(self.backward_old_data_block.write_back_parent_map)
            self.backward_old_data_block.reduce_forward()

        # get current data to write back between curr_op_space_start_ptr and curr_op_space_end_ptr
        for i in range(self.op_space_size):
            my_memory_idxs.append(self.curr_op_space_start_ptr)
            parent_memory_idxs.append(self.parent_map)
            self.curr_op_space_start_ptr = (self.curr_op_space_start_ptr + 1) % self.memory_size
            self.parent_map += 1

        # get old data to write back from forward_old_data_block
        for i in range(self.forward_old_data_block.size):
            my_memory_idxs.append(self.forward_old_data_block.start)
            parent_memory_idxs.append(self.forward_old_data_block.write_back_parent_map)
            self.forward_old_data_block.reduce_forward()

        self.reset_ptrs()
        return my_memory_idxs, parent_memory_idxs

    """
    "is_intersecting_forward" checks if the current mini memory would cause an internal or external
    collision when it is moved forward (i.e. one index to the right) in a positive delta prefetch. 
    There are 3 cases of collisions (2 external collisions where the current mini memory encroaches
    on the next mini memory, 1 internal collision where the current mini memory encroaches on its 
    old data)
        Case 1 (external): intersect the backward old data block of the next mini memory
        Case 2 (external): intersect the working region of the next mini memory
        Case 3 (internal): intersect its own forward old data block

    It returns a tuple of 4 values:
        a) boolean: did it intersect the next mini memory
        b) int: the index in the actual memory of the intersection
        c) int: the index in the parent memory of the intersection
        d) int: whether moving forward caused an internal or external intersection

    params:
        next_mini_memory:   the mini memory immediately right adjacent to the current mini memory in
                            the actual memory

    """
    def is_intersecting_forward(self, next_mini_memory):
        # ret_ptr denotes the index that the where current mini memory will encorach on, so need
        # to check if there is an intersection at ret_ptr
        ret_ptr = self.curr_op_space_end_ptr

        # Case 1
        is_intersecting, write_back_parent_map = next_mini_memory.backward_old_data_block.is_intersecting_forward(self.curr_op_space_end_ptr)
        if is_intersecting:
            next_mini_memory.backward_old_data_block.adjust_forward()
            self.forward_old_data_block.adjust_forward()
            return True, True, ret_ptr, write_back_parent_map, +1

        # Case 2
        if self.curr_op_space_end_ptr == next_mini_memory.curr_op_space_start_ptr:
            next_mini_memory.backward_old_data_block.adjust_forward()
            self.forward_old_data_block.adjust_forward()
            return True, True, ret_ptr, next_mini_memory.parent_map, +1

        # Case 3
        is_intersecting, write_back_parent_map = self.forward_old_data_block.is_intersecting_forward(self.curr_op_space_end_ptr)
        if is_intersecting:
            self.forward_old_data_block.adjust_forward()
            return False, False, ret_ptr, None, +0
        
        self.forward_old_data_block.adjust_forward()
        return False, True, ret_ptr, None, 0

    """
    "increment_ptrs" adjust the pointers of the mini memory in the forward direction (i.e. to the right)
    in the case of a positive delta prefetch
    params:
        is_static:  boolean determining whether the curr_op_space_start_ptr should stay in the same
                    place or if it should shift with the curr_op_space_end_ptr
        is_first:   bool indicating if this is the first mini memory touched in a prefetch
    """
    def increment_ptrs(self, is_static, is_first):
        if not is_static:
            self.curr_op_space_start_ptr = (self.curr_op_space_start_ptr + 1) % self.memory_size
            self.parent_map += 1
            # since the curr_op_space_start_ptr shifted right, the backward_old_data_block must
            # expand to encapsulated the data previously at the curr_op_space_start_ptr 
            if is_first: self.backward_old_data_block.should_increase = True
            self.backward_old_data_block.increase_forward()
        self.curr_op_space_end_ptr = (self.curr_op_space_end_ptr + 1) % self.memory_size

    """
    "is_intersecting_backward" checks if the current mini memory would cause an internal or external
    collision when it is moved backward (i.e. one index to the left) in a negative delta prefetch. 
    There are 3 cases of collisions (2 external collisions where the current mini memory encroaches
    on the next mini memory, 1 internal collision where the current mini memory encroaches on its 
    old data)
        Case 1 (external): intersect the forward old data block of the previous mini memory
        Case 2 (external): intersect the working region of the previous mini memory
        Case 3 (internal): intersect its own backward old data block

    It returns a tuple of 4 values:
        a) boolean: did it intersect the previous mini memory
        b) int: the index in the actual memory of the intersection
        c) int: the index in the parent memory of the intersection
        d) int: whether moving backward caused an internal or external intersection

    params:
        prev_mini_memory:   the mini memory immediately left adjacent to the current mini memory in
                            the actual memory

    """
    def is_intersecting_backward(self, prev_mini_memory):
        # ret_ptr denotes the index that the where current mini memory will encroach on, so need
        # to check if there is an intersection at ret_ptr
        ret_ptr = (self.curr_op_space_start_ptr - 1) % self.memory_size

        # Case 1
        is_intersecting, write_back_parent_map = prev_mini_memory.forward_old_data_block.is_intersecting_backward(self.curr_op_space_start_ptr)
        if is_intersecting:
            prev_mini_memory.forward_old_data_block.adjust_backward()
            self.backward_old_data_block.adjust_backward()
            return True, True, ret_ptr, write_back_parent_map, -1

        # Case 2
        if self.curr_op_space_start_ptr == prev_mini_memory.curr_op_space_end_ptr:
            prev_mini_memory.forward_old_data_block.adjust_backward()
            self.backward_old_data_block.adjust_backward()
            return True, True, ret_ptr, prev_mini_memory.parent_map + prev_mini_memory.op_space_size - 1, -1 

        # Case 3
        is_intersecting, write_back_parent_map = self.backward_old_data_block.is_intersecting_backward(self.curr_op_space_start_ptr)
        if is_intersecting:
            self.backward_old_data_block.adjust_backward()
            return False, False, ret_ptr, None, +0
    
        self.backward_old_data_block.adjust_backward()
        return False, True, ret_ptr, None, 0

    """
    "decrement_ptrs" adjust the pointers of the mini memory in the backward direction (i.e. to the left)
    in the case of a negative delta prefetch
    params:
        is_static:  boolean determining whether the curr_op_space_end_ptr should stay in the same
                    place or if it should shift with the curr_op_space_start_ptr
        is_first:   bool indicating if this is the first mini memory touched in a prefetch
    """
    def decrement_ptrs(self, is_static, is_first):
        if not is_static:
            self.curr_op_space_start_ptr = (self.curr_op_space_start_ptr - 1) % self.memory_size
            self.parent_map -= 1
            if is_first: self.forward_old_data_block.should_increase = True
            self.forward_old_data_block.increase_backward()
        self.curr_op_space_end_ptr = (self.curr_op_space_end_ptr - 1) % self.memory_size

    """
    "print" prints information about the mini memory
    """
    def print(self):
        print("\t Op Space Size:", self.op_space_size)
        print("\t Start:", self.curr_op_space_start_ptr)
        print("\t End:", self.curr_op_space_end_ptr)
        print("\t Backward Old Data Block")
        self.backward_old_data_block.print()
        print("\t Forward Old Data Block")
        self.forward_old_data_block.print()
        print()
