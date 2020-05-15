"""
Closed Form Solution for Read/Write Counts Without Parallelism
-- With parallelism is supported with "*_L1_parallel_calc_read_count" functions
#########################################################################################################

This is the mapping of timeloop to my dimension names:
P: output spatial coordinate
R: weight spatial coordinate
(P + R - 1): input spatial coordinate
C: channel index
K: filter index

level ordering:
memory closest to computation is L0, memory furthest away from computation is Ln

inverse level ordering:
memory closest to computation is Ln, memory furthest away from computation is L0

WeightMemory and OutputMemory have full prefetches everytime so no need to worry about delta prefetches

To determine the number of read counts at inverse_level n by inverse_level n-1:
    1) find full prefetch size of memory at inverse_level n-1
    2) examine loop tiling between inverse_level n and inverse_level n-1 memory
        a) find lowest dim that is in the memory's dependency set
        b) find product from lowest dim to highest dim of topmost loop tile
    3) multiply product of dims with full prefetch size
The result is the number of read counts for the first iteration of the entire convolution
To find total read counts, multiply by range of highest dim


InputMemory have both full and delta prefetches, so we need to analytically determine when a certain 
prefetch scheme is used. When the lowest dim of an InputMemory is "input", then there will always be
a delta prefetch when changing the val of a single dim ("weight" or "output") at a time. When changing
the val of the next dimension, there is the potential for crossover (delta prefetch) and if not, we do 
a full prefetch. When the lowest dim on an InputMemory is "channel", then there will never be a delta
prefetch and so will always have full prefetches.

To determine the number of read counts at inverse_level n by inverse_level n-1: 
    1) find full prefetch size of memory at inverse_level n-1
        a) determine what the lowest dim of memory at inverse_level n-1 is
        b) determine number of channels and spatial size of input space
    2) exmaine loop tiling between inverse_level n and n-1 memory
    3) if lowest dim == "channel"
        a) find "channel" dim and take product of all dims to highest dim of topmost loop tile
        b) multiply product of dims with full prefetch size
    4) if lowest dim == "input"
        a) we know that the lowest dim in the loop tiling is either "weight" or "output"
        b) determine the associated spatial size of the lowest dim (i.e. take product of range of all same dims in lower loop tilings)
            i) this is the delta size for each step in the lowest dim
        c) # of delta prefetches = lowest dim range - 1 (why -1: because first prefetch is always full prefetch)
        d) Case 1: if second lowest dim is either "weight" or "output"
            i) determine the delta when crossing over from lowest dim to second lowest dim
                a) delta = (second lowest dim spatial size) - (lowest dim spatial size * (lowest dim range - 1))
                b) if delta is less than input spatial size of inverse_level n-1 memory, then can do delta prefetch
                    i) read counts:
                       (
                        full prefetch + 
                        (channel spatial size) * (lowest dim range - 1) * (lowest dim spatial size) * (second lowest dim range - 1) +
                        (channel spatial size) * ((second lowest dim spatial size) - (lowest dim spatial size * (lowest dim range - 1))) * (second lowest dim range - 1)
                       ) * 
                       product of all dimensions from second dim in loop tiling to highest dim of topmost tile
                c) else, need to do full prefetch
                    i) read counts:
                       (
                        full prefetch + 
                        (channel spatial size * (lowest dim range - 1) * lowest dim spatial size)
                       ) * 
                       product of all dimensions from third (second lowest) dim in loop tiling to highest dim of topmost tile
        e) Case 2: if second lowest dim is "channel": need to do full prefetch when changing val of channel dim
            i) read counts:
               (
                full prefetch + 
                (channel spatial size) * (lowest dim range - 1) * (lowest dim spatial size)
               ) * 
               product of all dimensions from third (second lowest) dim in loop tiling to highest dim of topmost tile

If we are using write backs, then:
# of read counts to inverse_level n from inverse_level n+/-1 = # of write counts to inverse_level n-1 from inverse_level n+/-1

Total read counts at inverse_level n = # of read counts from inverse_level n-1 (prefetch) + # of read counts from inverse_level n+1 (writeback)
"""

def input_per_inverse_level_calc_full_prefetch_size(op_space_sizes_dict):
    return op_space_sizes_dict["channel"] * (op_space_sizes_dict["weight"] + op_space_sizes_dict["output"] - 1) 

def weight_per_inverse_level_calc_full_prefetch_size(op_space_sizes_dict):
    return op_space_sizes_dict["channel"] * op_space_sizes_dict["filter"] * op_space_sizes_dict["weight"]

def output_per_inverse_level_calc_full_prefetch_size(op_space_sizes_dict):
    return op_space_sizes_dict["filter"] * op_space_sizes_dict["output"]

def input_per_inverse_level_calc_read_count(inverse_level,
                                    loop_tiling_lst,
                                    op_space_sizes_dict,
                                    full_prefetch_size,
                                    upper_dim_product,
                                    dependency_set,
                                    lowest_dependency_dim):

    # Case 1: lowest dependency dim is channel
    if lowest_dependency_dim == "channel":

        for dim_dict in loop_tiling_lst[inverse_level]:
            dim_name, dim_range = list(dim_dict.items())[0]
            upper_dim_product *= dim_range
            if dim_name == "channel": break

        read_count = upper_dim_product * full_prefetch_size
        return read_count

    # Case 2: lowest dependency dim is input
    else:

        # get the two lowest dependency dims
        lowest_dependency_dim_names = [None, None]
        lowest_dependency_dim_ranges = [None, None]
        dependency_dim_count = 0
        for dim_dict in reversed(loop_tiling_lst[inverse_level]):
            dim_name, dim_range = list(dim_dict.items())[0]
            if dim_name in dependency_set:
                lowest_dependency_dim_names[dependency_dim_count] = dim_name
                lowest_dependency_dim_ranges[dependency_dim_count] = dim_range

                dependency_dim_count += 1
                if dependency_dim_count == 2: break

        # tracks whether the weight and output dims are adjacent in the loop tile
        if (lowest_dependency_dim_names[0] in {"weight", "output"}) and \
           (lowest_dependency_dim_names[1] in {"weight", "output"}):
            is_weight_output_adjacent = True
        else:
            is_weight_output_adjacent = False

        # determine delta prefetches for lowest dependency dim
        delta_prefetch_size = op_space_sizes_dict[lowest_dependency_dim_names[0]] * \
                              op_space_sizes_dict["channel"] * \
                              (lowest_dependency_dim_ranges[0]-1) * \
                              (lowest_dependency_dim_ranges[1])

        input_op_space_size = op_space_sizes_dict["weight"] + op_space_sizes_dict["output"] - 1

        # if weight and output are adjacent, then need to determine the crossover delta
        if is_weight_output_adjacent:
            crossover_delta = op_space_sizes_dict[lowest_dependency_dim_names[1]] - \
                              (op_space_sizes_dict[lowest_dependency_dim_names[0]] * (lowest_dependency_dim_ranges[0]-1))

        # channel was the second lowest dependency dim, so force a full prefetch
        else:
            crossover_delta = input_op_space_size

        # Case 2a: valid crossover delta
        if abs(crossover_delta) < input_op_space_size:
            crossover_delta_prefetch_size = abs(crossover_delta) * \
                                            op_space_sizes_dict["channel"] * \
                                            (lowest_dependency_dim_ranges[1]-1)

        # Case 2b: no valid crossover delta
        else:
            crossover_delta_prefetch_size = full_prefetch_size * \
                                            (lowest_dependency_dim_ranges[1]-1)

        total_prefetch_size = full_prefetch_size + delta_prefetch_size + crossover_delta_prefetch_size

        # get product of all higher dims
        for dim_dict in loop_tiling_lst[inverse_level]:
            dim_name, dim_range = list(dim_dict.items())[0]
            if dim_name == lowest_dependency_dim_names[1]: break
            upper_dim_product *= dim_range

        read_count = total_prefetch_size * upper_dim_product
        return read_count

def weight_output_per_inverse_level_calc_read_count(inverse_level,
                                            loop_tiling_lst,
                                            op_space_sizes_dict,
                                            full_prefetch_size, 
                                            upper_dim_product,
                                            dependency_set,
                                            lowest_dependency_dim):

    for dim_dict in loop_tiling_lst[inverse_level]:
        dim_name, dim_range = list(dim_dict.items())[0]
        upper_dim_product *= dim_range
        if dim_name == lowest_dependency_dim: break

    read_count = full_prefetch_size * upper_dim_product
    return read_count

def input_L1_parallel_calc_read_count(op_space_sizes_dict,
                                      full_prefetch_size,
                                      num_parallel_instances,
                                      upper_dim_product,
                                      static_dim_dict_lst,
                                      dependency_set,
                                      lowest_dependency_dim):

    # Case 1: lowest dependency dim is channel
    if lowest_dependency_dim == "channel":

        # find lowest dependency dim in static dims
        should_multiply = False
        for static_dim_dict in reversed(static_dim_dict_lst):
            dim_name, dim_range = list(static_dim_dict.items())[0]

            if dim_name in dependency_set: should_multiply = True
            if should_multiply: upper_dim_product *= dim_range

        read_count = full_prefetch_size * \
                     num_parallel_instances * \
                     upper_dim_product

        return read_count
    
    # Case 2: lowest dependency dim is input
    else:

        # find lowest dependency dim in static dims
        should_multiply = False
        for i, static_dim_dict in enumerate(reversed(static_dim_dict_lst)):
            dim_name, dim_range = list(static_dim_dict.items())[0]

            # have to do a full prefetch each time
            if dim_name == "channel": 

                for static_dim_dict in static_dim_dict_lst[:len(static_dim_dict_lst) - i]:
                    dim_name, dim_range = list(static_dim_dict.items())[0]
                    upper_dim_product *= dim_range

                read_count = full_prefetch_size * \
                             num_parallel_instances * \
                             upper_dim_product

                return read_count

            # can do a delta prefetch
            elif dim_name == "weight" or dim_name == "output":

                delta_prefetch_size = op_space_sizes_dict[dim_name] * \
                                      op_space_sizes_dict["channel"] * \
                                      (dim_range-1)

                for static_dim_dict in static_dim_dict_lst[:len(static_dim_dict_lst) - i - 1]:
                    dim_name, dim_range = list(static_dim_dict.items())[0]
                    upper_dim_product *= dim_range

                read_count = (full_prefetch_size + delta_prefetch_size) * \
                             num_parallel_instances * \
                             upper_dim_product

                return read_count

        # the entire loop tile is unrolled
        read_count = full_prefetch_size * num_parallel_instances * upper_dim_product
        return

def weight_output_L1_parallel_calc_read_count(op_space_sizes_dict,
                                              full_prefetch_size,
                                              num_parallel_instances,
                                              upper_dim_product,
                                              static_dim_dict_lst,
                                              dependency_set,
                                              lowest_dependency_dim):

    should_multiply = False
    for static_dim_dict in reversed(static_dim_dict_lst):
        dim_name, dim_range = list(static_dim_dict.items())[0]

        if dim_name in dependency_set: should_multiply = True
        if should_multiply: upper_dim_product *= dim_range

    read_count = full_prefetch_size * \
                 num_parallel_instances * \
                 upper_dim_product

    return read_count

# get the read count at a specific inverse_level
def per_inverse_level_calc_read_count_wrapper(inverse_level, 
                                      loop_tiling_lst,
                                      parallel_for_dims_set,
                                      dependency_set,
                                      per_inverse_level_calc_full_prefetch_size_func,
                                      per_inverse_level_calc_read_count_func,
                                      L1_parallel_calc_read_count_func):

    # L0_memory case is handled separately
    if inverse_level == len(loop_tiling_lst) - 1: return 0

    # get op space sizes for all dims
    op_space_sizes_dict = {"channel": 1, "filter": 1, "weight": 1, "output": 1}

    for loop_tile in loop_tiling_lst[inverse_level+1:]:
        for dim_dict in loop_tile:
            dim_name, dim_range = list(dim_dict.items())[0]
            op_space_sizes_dict[dim_name] *= dim_range

    upper_dim_product = 1
    for loop_tile in loop_tiling_lst[:inverse_level]:
        for dim_dict in loop_tile:
            dim_name, dim_range = list(dim_dict.items())[0]
            upper_dim_product *= dim_range

    # full prefetch size is specific to memory type
    full_prefetch_size = per_inverse_level_calc_full_prefetch_size_func(op_space_sizes_dict)

    # check if we need to parallel unroll this inverse_level
    # only applicable to L1 memory read counts because L0 memories are unrolled
    if (parallel_for_dims_set is not None) and (inverse_level == len(loop_tiling_lst)-2):
        # group non-unrolled dimensions in static_dim_dict_lst
        static_dim_dict_lst = []
        num_parallel_instances = 1
        L0_prefetch_tile = loop_tiling_lst[-2]
        for dim_dict in L0_prefetch_tile:
            dim_name, dim_range = list(dim_dict.items())[0]
            if dim_name in parallel_for_dims_set: num_parallel_instances *= dim_range
            else: static_dim_dict_lst.append(dim_dict)

    # determine the lowest dependency dim of memory type
    lowest_dependency_dim = None
    for dim_dict in reversed(loop_tiling_lst[inverse_level]):
        dim_name = list(dim_dict.keys())[0]
        if dim_name in dependency_set: 
            lowest_dependency_dim = dim_name
            break
    assert lowest_dependency_dim is not None

    read_count = per_inverse_level_calc_read_count_func(inverse_level, 
                                                loop_tiling_lst, 
                                                op_space_sizes_dict,
                                                full_prefetch_size,
                                                upper_dim_product,
                                                dependency_set,
                                                lowest_dependency_dim)

    # check if we need to parallel unroll this inverse_level
    # only applicable to L1 memory read counts because L0 memories are unrolled
    if (parallel_for_dims_set is not None) and (inverse_level == len(loop_tiling_lst)-2):

        # if no dependency dims are unrolled, then just scale prefetches by # of parallel instances
        if lowest_dependency_dim not in parallel_for_dims_set:
            read_count *= num_parallel_instances

        # some dependency dims are unrolled, so need to recalculate read count
        else:
            read_count = L1_parallel_calc_read_count_func(op_space_sizes_dict,
                                                          full_prefetch_size,
                                                          num_parallel_instances,
                                                          upper_dim_product,
                                                          static_dim_dict_lst,
                                                          dependency_set,
                                                          lowest_dependency_dim)
    return read_count

def all_levels_calc_read_count_wrapper(read_write_counts_lst, 
                                       loop_tiling_lst, 
                                       parallel_for_dims_set,
                                       dependency_set,
                                       per_inverse_level_calc_full_prefetch_size_func,
                                       per_inverse_level_calc_read_count_func,
                                       L1_parallel_calc_read_count_func):

    for inverse_level in range(len(loop_tiling_lst)):
        curr_inverse_level_read_write_count_dict = {"read count": 0, "write count": 0}

        # get forward read counts (read counts when inverse_level n memory is read by inverse_level n-1 memory)
        read_count = per_inverse_level_calc_read_count_wrapper(inverse_level, 
                                                      loop_tiling_lst,
                                                      parallel_for_dims_set,
                                                      dependency_set,
                                                      per_inverse_level_calc_full_prefetch_size_func,
                                                      per_inverse_level_calc_read_count_func,
                                                      L1_parallel_calc_read_count_func) 

        curr_inverse_level_read_write_count_dict["read count"] += read_count

        # get backward write counts (write counts when inverse_level n memory is written by inverse_level n+1 memory)
        if inverse_level != 0:
            parent_inverse_level_read_write_count_dict = read_write_counts_lst[-1]
            curr_inverse_level_read_write_count_dict["write count"] += parent_inverse_level_read_write_count_dict["read count"]

        read_write_counts_lst.append(curr_inverse_level_read_write_count_dict)

def all_levels_calc_write_count_wrapper(read_write_counts_lst, write_backs_set, memory_type):
    # check if write back is enabled
    if memory_type in write_backs_set:
        for inverse_level in range(len(read_write_counts_lst)):
            curr_inverse_level_read_write_count_dict = read_write_counts_lst[inverse_level]
            # if not lowest memory, then can evaluate write backs
            if inverse_level != len(read_write_counts_lst)-1:
                child_inverse_level_read_write_count_dict = read_write_counts_lst[inverse_level+1]
                # in writeback, curr inverse_level is written back to with whatever it wrote to child inverse_level initially
                curr_inverse_level_read_write_count_dict["write count"] += child_inverse_level_read_write_count_dict["write count"]
                # in writeback, child inverse_level is read from with whatever it was written to initially
                child_inverse_level_read_write_count_dict["read count"] += child_inverse_level_read_write_count_dict["write count"]

def L0_load_store_read_write_count(read_write_counts_lst, loop_tiling_lst, memory_type):
    upper_dim_product = 1
    for loop_tile in loop_tiling_lst:
        for dim_dict in loop_tile:
            dim_name, dim_range = list(dim_dict.items())[0]
            upper_dim_product *= dim_range

    read_count = upper_dim_product
    read_write_counts_lst[-1]["read count"] += read_count

    # for L0 output memory, it is written to every innermost loop iteration by store operation
    # since L0 memory is never read from apart from load operation, can just increment write count by read coutn
    # # of store operations = # of load operations
    if memory_type == "output":
        read_write_counts_lst[-1]["write count"] += read_count


def read_write_count_analysis(loop_tiling_lst, parallel_for_dims_set, write_backs_set):

    input_read_write_counts_lst = []
    weight_read_write_counts_lst = []
    output_read_write_counts_lst = []

    all_levels_calc_read_count_wrapper(input_read_write_counts_lst, 
                                       loop_tiling_lst, 
                                       parallel_for_dims_set,
                                       {"channel", "weight", "output"},
                                       input_per_inverse_level_calc_full_prefetch_size,
                                       input_per_inverse_level_calc_read_count,
                                       input_L1_parallel_calc_read_count)

    all_levels_calc_read_count_wrapper(weight_read_write_counts_lst, 
                                       loop_tiling_lst, 
                                       parallel_for_dims_set,
                                       {"channel", "filter", "weight"},
                                       weight_per_inverse_level_calc_full_prefetch_size,
                                       weight_output_per_inverse_level_calc_read_count,
                                       weight_output_L1_parallel_calc_read_count)

    all_levels_calc_read_count_wrapper(output_read_write_counts_lst, 
                                       loop_tiling_lst, 
                                       parallel_for_dims_set,
                                       {"filter", "output"},
                                       output_per_inverse_level_calc_full_prefetch_size,
                                       weight_output_per_inverse_level_calc_read_count,
                                       weight_output_L1_parallel_calc_read_count)

    all_levels_calc_write_count_wrapper(input_read_write_counts_lst, write_backs_set, "input")
    all_levels_calc_write_count_wrapper(weight_read_write_counts_lst, write_backs_set, "weight")
    all_levels_calc_write_count_wrapper(output_read_write_counts_lst, write_backs_set, "output")

    L0_load_store_read_write_count(input_read_write_counts_lst, loop_tiling_lst, "input")
    L0_load_store_read_write_count(weight_read_write_counts_lst, loop_tiling_lst, "weight")
    L0_load_store_read_write_count(output_read_write_counts_lst, loop_tiling_lst, "output")

    return {"input":  input_read_write_counts_lst,
            "weight": weight_read_write_counts_lst,
            "output": output_read_write_counts_lst}