import numpy as np
import copy
import math
import random

class Queue:

    """
    "__init__" initializes a queue
    params:
    queue_size: size of the queue (i.e. how many elements it can maximally hold at a time)
                if queue_size is None, then the queue is unbounded (i.e. no maximum size)
    """
    def __init__(self, queue_size):
        # queue is physically implemented as an array
        self.queue = []
        # maximum size of the queue
        self.queue_size = queue_size
        # number of elements currently in the queue
        self.curr_size = 0

    """
    "clear" resets the queue
    params: None
    """
    def clear(self):
        self.queue = []
        self.curr_size = 0
    
    """
    "enque" appends data to the buffer
    params:
        val: the data to append to the buffer
    """
    def enque(self, val):
        # check if the buffer can support an additional data entry
        if (self.queue_size is not None) and (self.curr_size >= self.queue_size):
            print("Error: queue is full, cannot enque any elements")
            assert False
        # add val to the buffer and update the number of elements
        self.curr_size += 1
        self.queue.append(val)

    """
    "deque" removes data from the queue
    params: None
    """
    def deque(self):
        # check if there is data in the buffer to remove
        if self.curr_size <= 0:
            print("Error: buffer is empty, cannot deque any elements")
            assert False
        # remove data from the buffer and update the number of elements
        ret_val = self.queue.pop(0)
        self.curr_size -= 1
        return ret_val

    """
    "is_empty" checks if the buffer is empty
    params: None
    """
    def is_empty(self):
        return self.curr_size == 0

    def extend_max_size(self, queue_size):
        self.queue_size = queue_size

    def get_max_size(self):
        return self.queue_size

    def __len__(self):
        return self.curr_size
    
    """
    "__repr__" prints the data in the buffer
    params: None
    """
    def __repr__(self):
        return str(self.queue)

"""
"Buffer" functions as a FIFO queue to handle communication between memories at adjacent levels
where a buffer holds the data to be transmitted. Two buffers are implemented together to form a 
double-buffered system between memories
"""

class Buffer(Queue):
    """
    "__init__" initializes a buffer
    params:
    writer: instance of the Memory class that can write to a buffer
    reader: instance of the Memory class that can read from a buffer
    buf_size: size of the buffer (i.e. how many elements it can maximally hold at a time)
    """
    def __init__(self, writer, reader, buf_size):
        self.writer = writer
        self.reader = reader
        super().__init__(buf_size)

    
    """
    "send_data_parent_to_child" handles the memory transaction of a parent memory sending data
    to its child memory when the child memory is prefetching data
    params:
        read_request:   dictionary containing information about what index in the memory array data
                        is being read from in the parent
    """
    def send_data_parent_to_child(self, read_request):
        # the parent is the writer
        self.writer.write_forward(read_request)
    
    """
    "send_data_child_to_parent" handles the memory transaction of a child memory sending data
    to its parent memory when the child memory is writing back data
    params:
        write_request:  dictionary containing information about what index in the parent memory array 
                        is being written in
    """
    def send_data_child_to_parent(self, write_request):
        # the parent is the reader
        self.reader.read_forward(write_request)


class Trace_Queue(Queue):

    def __init__(self):
        super().__init__(None)

