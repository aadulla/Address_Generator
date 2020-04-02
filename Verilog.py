import numpy as np
import copy
import math
import random

from Queue import Queue

class Buffer_Verilog(Queue):
    def __init__(self, buf_size):
        super().__init__(buf_size)

        # pinout
        self.data = 0
        self.read_enable = 0
        self.write_enable = 0

    def set_read_enable(self):
        self.read_enable = 1

    def set_write_enable(self):
        self.write_enable = 1

    def set_data(self, val):
        self.data = val

    def get_data(self):
        return self.data

    def clear_pins(self):
        self.data = 0
        self.read_enable = 0
        self.write_enable = 0

    def clk_event(self):
        if self.read_enable == 1:
            self.data = self.deque()
        elif self.write_enable == 1:
            self.enque(self.data)


class Processing_Element_Verilog:
    def __init__(self):

        # pinout
        self.enable = 0
        self.input_val = 0
        self.weight_val = 0
        self.output_val = 0

    def set_enable(self):
        self.enable = 1

    def set_input(self, input_val):
        self.input_val = input_val

    def set_weight(self, weight_val):
        self.weight_val = weight_val

    def set_output(self, output_val):
        self.output_val = output_val

    def get_output(self):
        return self.output_val

    def clear_pins(self):
        self.enable = 0
        self.input_val = 0
        self.weight_val = 0
        self.output_val = 0

    def clk_event(self):
        if self.enable == 1:
            self.output_val += self.input_val * self.weight_val


class Memory_Verilog:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory_array = [None]*self.memory_size
        self.read_count = 0
        self.write_count = 0

        # pinout
        self.read_enable = 0
        self.write_enable = 0
        self.address = 0
        self.data = 0

    def initialize(self, data):

        def flatten(lst):
            if lst == []:
                return lst
            if isinstance(lst[0], list):
                return flatten(lst[0]) + flatten(lst[1:])
            return lst[:1] + flatten(lst[1:])
        
        flat_data = flatten(copy.copy(data))
        for i, val in enumerate(flat_data):
            self.memory_array[i] = val

    def set_read_enable(self):
        self.read_enable = 1

    def set_write_enable(self):
        self.write_enable = 1

    def set_address(self, address):
        self.address = address

    def set_data(self, val):
        self.data = val

    def get_data(self):
        return self.data

    def clear_pins(self):
        self.address = 0
        self.read_enable = 0
        self.write_enable = 0
        self.data = 0

    def clk_event(self):
        if self.read_enable == 1:
            self.read_count += 1
            self.data = self.memory_array[self.address]
        elif self.write_enable == 1:
            self.write_count += 1
            self.memory_array[self.address] = self.data

    def print(self):
        print(self.memory_array)

class Controller_Verilog:
    def __init__(self, hierarchy_manager, memory_trace_queue, memory_type):
        self.hierarchy_manager = hierarchy_manager
        self.memory_trace_queue = memory_trace_queue
        self.memory_type = memory_type
        self.clk_count = 0

    # buffer is read from
    def read_buffer(self, buffer_verilog):
        buffer_verilog.set_read_enable()
        buffer_verilog.clk_event()
        data = buffer_verilog.get_data()
        buffer_verilog.clear_pins()
        return data

    # buffer is written to
    def write_buffer(self, buffer_verilog, data):
        buffer_verilog.set_data(data)
        buffer_verilog.set_write_enable()
        buffer_verilog.clk_event()
        buffer_verilog.clear_pins()

    # memory is read from
    def read_memory(self, memory_verilog, address):
        memory_verilog.set_address(address)
        memory_verilog.set_read_enable()
        memory_verilog.clk_event()
        data = memory_verilog.get_data()
        memory_verilog.clear_pins()
        return data

    # memory is written to
    def write_memory(self, memory_verilog, address, data):
        memory_verilog.set_address(address)
        memory_verilog.set_data(data)
        memory_verilog.set_write_enable()
        memory_verilog.clk_event()
        memory_verilog.clear_pins()

    def read_PE(self, PE):
        data = PE.get_output()
        PE.clear_pins()
        return data

    def write_PE(self, PE, data):
        if self.memory_type == "input":
            PE.set_input(data)
        if self.memory_type == "weight":
            PE.set_weight(data)
        if self.memory_type == "output":
            PE.set_output(data)

    def execute_command(self, command, PE):
        level   = command["level"]
        op      = command["op"]
        address = command["address"]

        """
        setup appropriate inputs to pins on verilog memory and buffer
        Case 1: read forward (writeback)
            - parent memory is being written to by child memory
        Case 2: read backward (prefetch)
            - child memory is being written to by parent memory
        Case 3: write forward (prefetch)
            - parent memory is being read from by child memory
        Case 4: write backward (writeback)
            - child memory is being read from by parent memory
        Case 5: load
            - L0 child memory is read from
        Case 6: store
            - L0 child memory is written to
        """

        # Case 1
        if op == "read forward":
            buffer_type = op
            buffer_verilog = self.hierarchy_manager.get_buffer_verilog(level, buffer_type)
            data = self.read_buffer(buffer_verilog)

            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            self.write_memory(memory_verilog, address, data)

            # policy = trace["policy"]
            # if policy == "overwrite":
            #   self.write_memory(memory_verilog, address, data)
            # else: 
            #   raise NotImplementedError

        # Case 2
        if op == "read backward":
            buffer_type = op
            buffer_verilog = self.hierarchy_manager.get_buffer_verilog(level, buffer_type)
            data = self.read_buffer(buffer_verilog)

            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            self.write_memory(memory_verilog, address, data)

        # Case 3
        if op == "write forward":
            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            data = self.read_memory(memory_verilog, address)

            buffer_type = op
            buffer_verilog = self.hierarchy_manager.get_buffer_verilog(level, buffer_type)
            data = self.write_buffer(buffer_verilog, data)

        # Case 4
        if op == "write backward":
            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            data = self.read_memory(memory_verilog, address)

            buffer_type = op
            buffer_verilog = self.hierarchy_manager.get_buffer_verilog(level, buffer_type)
            data = self.write_buffer(buffer_verilog, data)

        # Case 5
        if op == "load":
            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            data = self.read_memory(memory_verilog, address)
            self.write_PE(PE, data)

        # Case 6
        if op == "store":
            memory_verilog = self.hierarchy_manager.get_memory_verilog(level)
            data = self.read_PE(PE)
            self.write_memory(memory_verilog, address, data)

        self.clk_count += 1

    # keep iterating through commands until reaching a barrier sync or end of trace
    def execute_trace(self, PE, stop_conditions_dict):
        while (True):
            command = self.memory_trace_queue.deque()
            if command == "BARRIER" or command == "END": 
                stop_conditions_dict[self.memory_type] = command
                return
            self.execute_command(command, PE)

    def print_stats(self):
        print("Total Clk Count:", self.clk_count)
        print("#"*100)
        self.hierarchy_manager.print_stats()