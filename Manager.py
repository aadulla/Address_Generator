import numpy as np
import copy
import math
import random
import threading

from Verilog import *

# this is not a physical verilog implementation, this is just an abstraction
class Hierarchy_Manager:
    def __init__(self, hierarchy_lst):
        self.num_levels = len(hierarchy_lst)

        # create disctionary to store verilog memories at each level
        self.memory_verilog_dict = dict()
        for i, memory in enumerate(hierarchy_lst):
            memory_size = memory.memory_size
            memory_verilog = Memory_Verilog(memory_size)
            level = self.num_levels - i - 1
            self.memory_verilog_dict.update({level: memory_verilog})

        # create dictionary to store pointers to forward and backward verilog buffers
        # for the verilog memories
        self.buffer_verilog_dict = dict()
        for i in range(self.num_levels):

            """ 
            dictionary entry:
            {<level (int)>: {
                            "read forward"  : <read forward buffer   (ptr)>,
                            "read backward" : <read backward buffer  (ptr>)>,
                            "write forward" : <write forward buffer  (ptr>)>,
                            "write backward": <write backward buffer (<ptr)>,
                            }
            """
            self.buffer_verilog_dict.update({i: {
                                                "read forward": None, 
                                                "read backward": None, 
                                                "write forward": None, 
                                                "write backward": None,
                                                }
                                            })

        # create verilog buffers in forward direction (parent->child)
        for i in range(0, len(hierarchy_lst)-1, 1):
            parent = hierarchy_lst[i]
            child = hierarchy_lst[i+1]
            buf_size = child.op_space_size
            buffer_verilog = Buffer_Verilog(buf_size=buf_size)

            parent_level = self.num_levels - i - 1
            child_level = parent_level - 1
            self.buffer_verilog_dict[parent_level]["write forward"] = buffer_verilog
            self.buffer_verilog_dict[child_level]["read backward"] = buffer_verilog
            
        # create verilog buffers in backward direction (child->parent)
        for i in range(len(hierarchy_lst)-1, 0, -1):
            parent = hierarchy_lst[i-1]
            child = hierarchy_lst[i]
            buf_size = child.op_space_size
            buffer_verilog = Buffer_Verilog(buf_size=buf_size)

            parent_level = self.num_levels - i
            child_level = parent_level - 1
            self.buffer_verilog_dict[parent_level]["read forward"] = buffer_verilog
            self.buffer_verilog_dict[child_level]["write backward"] = buffer_verilog

    def get_memory_verilog(self, level):
        if level == -1: level = self.num_levels - 1
        memory_verilog = self.memory_verilog_dict[level]
        return memory_verilog

    def get_buffer_verilog(self, level, buffer_type):
        buffer_verilog = self.buffer_verilog_dict[level][buffer_type]
        return buffer_verilog

    def print_stats(self):
        for i in range(self.num_levels-1, -1, -1):
            print("Level:", i)
            memory_verilog = self.get_memory_verilog(i)
            read_count = memory_verilog.read_count
            print("Read Count:", read_count)
            write_count = memory_verilog.write_count
            print("Write Count:", write_count)
            print("#"*100)


class Controller_Manager:
    def __init__(self, 
                 input_memory_trace_queue, weight_memory_trace_queue, output_memory_trace_queue,
                 input_hierarchy_lst, weight_hierarchy_lst, output_hierarchy_lst,
                 input_data, weight_data, output_data):

        self.input_memory_trace_queue  = input_memory_trace_queue
        self.weight_memory_trace_queue = weight_memory_trace_queue
        self.output_memory_trace_queue = output_memory_trace_queue

        self.input_hierarchy_manager  = Hierarchy_Manager(input_hierarchy_lst)
        self.weight_hierarchy_manager = Hierarchy_Manager(weight_hierarchy_lst)
        self.output_hierarchy_manager = Hierarchy_Manager(output_hierarchy_lst)

        base_input_memory_verilog  = self.input_hierarchy_manager.get_memory_verilog(-1)
        base_input_memory_verilog.initialize(input_data)
        base_weight_memory_verilog = self.weight_hierarchy_manager.get_memory_verilog(-1)
        base_weight_memory_verilog.initialize(weight_data)
        base_output_memory_verilog = self.output_hierarchy_manager.get_memory_verilog(-1)
        base_output_memory_verilog.initialize(output_data)

        self.input_controller  = Controller_Verilog(self.input_hierarchy_manager,  
                                                    self.input_memory_trace_queue, 
                                                    "input")
        self.weight_controller = Controller_Verilog(self.weight_hierarchy_manager, 
                                                    self.weight_memory_trace_queue,
                                                    "weight")
        self.output_controller = Controller_Verilog(self.output_hierarchy_manager, 
                                                    self.output_memory_trace_queue,
                                                    "output")

        self.PE = Processing_Element_Verilog()

    def run(self):
        while (True):

            stop_conditions_dict = {"input": None, "weight": None, "output": None}
            input_thread  = threading.Thread(target=self.input_controller.execute_trace, 
                                             args=(self.PE, stop_conditions_dict))
            weight_thread = threading.Thread(target=self.weight_controller.execute_trace,
                                             args=(self.PE, stop_conditions_dict))
            output_thread = threading.Thread(target=self.output_controller.execute_trace,
                                             args=(self.PE, stop_conditions_dict))

            # input_stop_condition = self.input_controller.execute_trace(self.PE)
            # weight_stop_condition = self.weight_controller.execute_trace(self.PE)
            # output_stop_condition = self.output_controller.execute_trace(self.PE)

            input_thread.start()
            weight_thread.start()
            output_thread.start()

            input_thread.join()
            weight_thread.join()
            output_thread.join()

            input_stop_condition  = stop_conditions_dict["input"]
            weight_stop_condition = stop_conditions_dict["weight"]
            output_stop_condition = stop_conditions_dict["output"]

            assert input_stop_condition == weight_stop_condition == output_stop_condition

            if input_stop_condition == "BARRIER":
                self.PE.set_enable()
                self.PE.clk_event()
            elif input_stop_condition == "END":
                break

        input_trace_queue_empty  = self.input_memory_trace_queue.is_empty()
        weight_trace_queue_empty = self.weight_memory_trace_queue.is_empty()
        output_trace_queue_empty = self.output_memory_trace_queue.is_empty()

        assert input_trace_queue_empty
        assert weight_trace_queue_empty
        assert output_trace_queue_empty

    def get_base_input(self):
        base_input_memory_verilog  = self.input_hierarchy_manager.get_memory_verilog(-1)
        base_input_memory_array  = base_input_memory_verilog.memory_array
        return base_input_memory_array

    def get_base_weight(self):
        base_weight_memory_verilog  = self.weight_hierarchy_manager.get_memory_verilog(-1)
        base_weight_memory_array  = base_weight_memory_verilog.memory_array
        return base_weight_memory_array

    def get_base_output(self):
        base_output_memory_verilog  = self.output_hierarchy_manager.get_memory_verilog(-1)
        base_output_memory_array  = base_output_memory_verilog.memory_array
        return base_output_memory_array

    def print_stats(self):
        print("INPUT MEMORY HIERARCHY")
        self.input_controller.print_stats()
        print()

        print("WEIGHT MEMORY HIERARCHY")
        self.weight_controller.print_stats()
        print()

        print("OUTPUT MEMORY HIERARCHY")
        self.output_controller.print_stats()
        print()







