import numpy as np
import copy
import math
import random

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
"DataBlock" represents a block of data inside a MiniMemory instance that is stale
"""
class DataBlock:
    """
    "__init__" initializes a data block
    params:
        initial:        the index in the actual memory of where the data block initially starts

        memory_size:    the size of the actual memory

    """
    def __init__(self, initial, memory_size):
        self.initial = initial
        
        # initialize a size 0 data block
        self.start = self.initial
        self.end = self.initial
        self.size = 0

        # pointer attached to self.start that maps where self.start is in the current memory to
        # where it is in the parent memory
        self.write_back_parent_map = 0

        self.memory_size = memory_size

        # determines whether a data block should expand itself or not
        self.should_increase = True
        
    """
    "reset" resets the datablock
    params: None
    """
    def reset(self):
        self.start = self.initial
        self.end = self.initial
        self.write_back_parent_map = 0
        self.size = 0

    """
    "is_intersecting_forward" determines whether the start of the datablock points to the same
    location/index in the actual memory of a passed in pointer
    params: 
        ptr: a pointer to the location in the actual memory 
    """
    def is_intersecting_forward(self, ptr):
        # check if the data block is valid (i.e. not empty/non-zero size)
        if self.size != 0:
            # check if ptr points to the same place as the start of the datablock
            if ptr == self.start:
                # ret_parent_map is the mapping to the index at self.start
                ret_parent_map = self.write_back_parent_map
                return True, ret_parent_map
            else:
                return False, None
        else: return False, None

    """
    "is_intersecting_backward" determines whether the end of the datablock points to the same
    location/index in the actual memory of a passed in pointer
    params: 
        ptr: a pointer to the location in the actual memory 
    """
    def is_intersecting_backward(self, ptr):
        # check if the data block is valid (i.e. not empty/non-zero size)
        if self.size != 0:
            # check if ptr points to the same place as the start of the datablock
            if ptr == self.end:
                # ret_parent_map is the mapping to the index before self.end
                ret_parent_map = self.write_back_parent_map + self.size - 1
                return True, ret_parent_map
            else: return False, None
        else: return False, None
    
    """
    "increase_forward" expands the datablock in the forward direction (i.e. it grows towards the right)
    params: None
    """
    def increase_forward(self):
        if self.should_increase:
            self.size += 1
            # move self.end one index right
            self.end = (self.start + self.size) % self.memory_size

    """
    "increase_backward" expands the datablock in the forward direction (i.e. it grows towards the left)
    params: None
    """
    def increase_backward(self):
        if self.should_increase:
            self.size += 1
            # move self.start one index left
            self.start = (self.end - self.size) % self.memory_size
    
    """
    "reduce_forward" shrinks the datablock in the forward direction (i.e. it shirnks towards the right)
    params: None
    """
    def reduce_forward(self):
        self.size -= 1
        # move self.start one index right
        self.start = (self.start + 1) % self.memory_size
        self.write_back_parent_map += 1

    """
    "reduce_backward" shrinks the datablock in the forward direction (i.e. it shirnks towards the right)
    params: None
    """  
    def reduce_backward(self):
        self.size -= 1
        # move self.end one index left
        self.end = (self.end - 1) % self.memory_size

    """
    "shift_right" shifts the entire data block one index to the right
    params: None
    """
    def shift_right(self):
        self.start = (self.start + 1) % self.memory_size
        self.end = (self.end + 1) % self.memory_size
        self.write_back_parent_map += 1

    """
    "shift_left" shifts the entire data block one index to the left
    params: None
    """
    def shift_left(self):
        self.start = (self.start - 1) % self.memory_size
        self.end = (self.end - 1) % self.memory_size
        self.write_back_parent_map -= 1

    """
    "adjust_forward" adjusts the datablock in the forward direction as a result of the previous
    MiniMemory encroaching on the datablock
    params: None
    """
    def adjust_forward(self):
        # if the datablock is empty, shift it to the right
        if self.size == 0:
            self.shift_right()
            self.should_increase = False
        # if the datablock is non-empty, shrink the datablock from the start
        elif self.size != 0:
            self.reduce_forward()
            self.should_increase = True

    """
    "adjust_forward" adjusts the datablock in the backward direction as a result of the next
    MiniMemory encroaching on the datablock
    params: None
    """
    def adjust_backward(self):
        # if the datablock is empty, shift it to the left
        if self.size == 0:
            self.shift_left()
            self.should_increase = False
        # if the datablock is non-empty, shrink the datablock from the end
        elif self.size != 0:
            self.reduce_backward()
            self.should_increase = True

    """
    "print" prints information about the datablock
    """
    def print(self):
        print("Start Ptr:", self.start)
        print("End Ptr:", self.end)
        print("Parent Map:", self.write_back_parent_map)
        print("Initial:", self.initial)
        print("Size:", self.size)