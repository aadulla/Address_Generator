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

                def memory_prefetch_wrapper(memory,
                                            prefetch_level, 
                                            curr_level, 
                                            controller_loop_counters, 
                                            loop_order_lst):

                    if prefetch_level == curr_level:
                        memory.prefetch(None, 
                                        controller_loop_counters,
                                        loop_order_lst)


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

                    memory_prefetch_wrapper(weight_memory, 
                                            weight_prefetch_level,
                                            0,
                                            loop_counters[hierarchy_index-1],
                                            curr_loop_order_lst)

                    memory_prefetch_wrapper(output_memory, 
                                            output_prefetch_level,
                                            0,
                                            loop_counters[hierarchy_index-1],
                                            curr_loop_order_lst)

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

                        memory_prefetch_wrapper(weight_memory, 
                                                weight_prefetch_level,
                                                1,
                                                loop_counters[hierarchy_index-1],
                                                curr_loop_order_lst)

                        memory_prefetch_wrapper(output_memory, 
                                                output_prefetch_level,
                                                1,
                                                loop_counters[hierarchy_index-1],
                                                curr_loop_order_lst)
                        
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

                            memory_prefetch_wrapper(weight_memory, 
                                                    weight_prefetch_level,
                                                    2,
                                                    loop_counters[hierarchy_index-1],
                                                    curr_loop_order_lst)

                            memory_prefetch_wrapper(output_memory, 
                                                    output_prefetch_level,
                                                    2,
                                                    loop_counters[hierarchy_index-1],
                                                    curr_loop_order_lst)

                            curr_level = 3
                            if curr_level == first_prefetch_level: start = True

                            loop_counters[hierarchy_index-1][dim_3] = 0

                            for idx_3 in range(tilings_dict[dim_3][hierarchy_index - 1]):
                                dim_idxs[dim_3] = idx_3
                                dim_totals[dim_3] = 1
                                for tile_3 in tilings_dict[dim_3][hierarchy_index:]:
                                    dim_totals[dim_3] *= tile_3
                                loop_counters[hierarchy_index-1][dim_3] = idx_3

                                memory_prefetch_wrapper(weight_memory, 
                                                        weight_prefetch_level,
                                                        3,
                                                        loop_counters[hierarchy_index-1],
                                                        curr_loop_order_lst)

                                memory_prefetch_wrapper(output_memory, 
                                                        output_prefetch_level,
                                                        3,
                                                        loop_counters[hierarchy_index-1],
                                                        curr_loop_order_lst)

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
                                                          curr_loop_order_lst)
    
                                else:
                                    # perform a delta prefetch
                                    is_fresh_prefetch = input_memory.prefetch(input_delta, 
                                                          loop_counters[hierarchy_index-1],
                                                          curr_loop_order_lst)

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
                    input_memory.write_back_op_space()
                if "weight" in write_backs:
                    weight_memory.write_back_op_space()
                if "output" in write_backs:
                    output_memory.write_back_op_space()
                    
                return