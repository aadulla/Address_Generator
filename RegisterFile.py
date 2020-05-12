class RegisterFile:
	def __init__(self, loop_counter_asst, input_memory, weight_memory, output_memory):
		self.loop_counter_asst = loop_counter_asst
		self.input_memory = input_memory
		self.weight_memory = weight_memory
		self.output_memory = output_memory

		self.input_prev_block_start, self.input_curr_block_start = 0, 0

