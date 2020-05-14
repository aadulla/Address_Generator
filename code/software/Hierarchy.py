import numpy as np
import copy
import math
import random
import sys
sys.path.append('../lib/')

from lib.Queue import Buffer

class Hierarchy:
    def __init__(self, hierarchy_lst):
        self.hierarchy_lst = hierarchy_lst
        
        # link forward direction buffers
        for i in range(0, len(self.hierarchy_lst)-1, 1):
            parent = self.hierarchy_lst[i]
            child = self.hierarchy_lst[i+1]
            # the buffer size is hardcoded at the child's memory size. the buffer size
            # can be upper bounded by the child's prefetch size
            buf_size = child.memory_size
            buffer = Buffer(writer=parent, reader=child, buf_size=buf_size)
            parent.write_forward_buffer = child.read_backward_buffer = buffer
            
        # link backward direction buffers
        for i in range(len(self.hierarchy_lst)-1, 0, -1):
            child = self.hierarchy_lst[i]
            parent = self.hierarchy_lst[i-1]
            # the buffer size is hardcoded at the child's memory size. the buffer size
            # can be upper bounded by the child's prefetch size
            buf_size = child.memory_size
            buffer = Buffer(writer=child, reader=parent, buf_size=buf_size)
            child.write_backward_buffer = parent.read_forward_buffer = buffer

    def parallel_unroll(self, num_PEs):
        L0_memory = self.hierarchy_lst[-1]

        # save pointers to L1 buffers
        L1_write_backward_buffer = L0_memory.write_backward_buffer
        L1_read_backward_buffer  = L0_memory.read_backward_buffer

        # set buffers to None so do not deepcopy buffers when unrolling memories
        L0_memory.write_backward_buffer = None
        L0_memory.read_backward_buffer = None

        # create individual L0 memories for each PE
        unrolled_L0_memories = [copy.deepcopy(L0_memory) for i in range(num_PEs)]
        for memory in unrolled_L0_memories:
            memory.write_backward_buffer = L1_write_backward_buffer
            memory.read_backward_buffer  = L1_read_backward_buffer

        self.hierarchy_lst[-1] = unrolled_L0_memories

        del L0_memory

    def __len__(self):
        return len(self.hierarchy_lst)
    
    def __getitem__(self, index):
        return self.hierarchy_lst[index]
    
    def calc_cost(self):
        total_cost = 0
        for memory in self.hierarchy_lst:
            if type(memory) == list:
                for parallel_memory in memory:
                    total_cost += parallel_memory.calc_cost()
            else: 
                total_cost += memory.calc_cost()
        return total_cost
    
    def print(self):
        for memory in self.hierarchy_lst:
            memory.print()
            print("="*30)
    
    def print_stats(self):
        print("Total Cost:", self.calc_cost())
        print("#"*100)
        for i, memory in enumerate(self.hierarchy_lst):
            print("Level: ", len(self.hierarchy_lst) - i)
            if type(memory) == list:
                for parallel_memory in memory:
                    parallel_memory.print_stats("\t")
                    print("#"*100)
            else:
                memory.print_stats()
            print("#"*100)
        print()