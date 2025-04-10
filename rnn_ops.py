import tensorflow as tf
from tensorflow.python.framework import constant_op, dtypes, ops, tensor_shape
from tensorflow.python.ops import array_ops, math_ops, tensor_array_ops
from tensorflow.python.util import nest


# Helper to get shape for TensorArray element_shape
def _maybe_tensor_shape_from_tensor(t_shape):
    """Converts a shape Tensor or tuple/list/TensorShape to a TensorShape."""
    if isinstance(t_shape, ops.Tensor):
        # Use tf.get_static_value if possible, otherwise keep dynamic
        static_val = tf.get_static_value(t_shape)
        if static_val is not None:
            return tensor_shape.TensorShape(static_val)
        else:
            # Cannot determine static shape. Try to get rank and return partial shape.
            try:
                 rank = t_shape.shape[0] # Rank is the length of the shape tensor
                 if rank is not None:
                      return tensor_shape.TensorShape([None] * rank) # Return shape with Nones
                 else:
                      # If rank cannot be determined statically, cannot create shape.
                      raise ValueError(f"Cannot determine rank statically from tensor: {t_shape}")
            except: # Catch potential errors if t_shape.shape doesn't behave as expected
                 raise ValueError(f"Cannot determine static shape or rank from tensor: {t_shape}")

    else:
        # Input is already list/tuple/TensorShape
        return tensor_shape.TensorShape(t_shape)


def raw_rnn(cell, loop_fn, constants, parallel_iterations=32, swap_memory=False, scope=None):
    """
    TF2-compatible adaptation of TF1's raw_rnn for custom RNN loops.
    Args:
        cell: A Keras RNN cell instance (e.g., LSTMAttentionCell).
        loop_fn: A callable with signature:
                 loop_fn(time, cell_output, cell_state, loop_state) ->
                     (finished, next_input, next_cell_state, emit_output, next_loop_state)
                 Note: `cell_output` is the raw output from the cell (e.g., h3),
                       `cell_state` is the full state tuple *after* the RNN step.
        constants: Data passed unchanged to the cell's `call` method at each step.
        parallel_iterations: Passed to `tf.while_loop`. Defaults to 32.
        swap_memory: Passed to `tf.while_loop`. Defaults to False.
        scope: Name scope for operations.
    Returns:
        Tuple: (all_states, all_outputs, final_state) where
               all_states: Nested structure matching cell state, stacked over time.
               all_outputs: Nested structure matching emit_output, stacked over time.
               final_state: The final state tuple of the cell.
    """
    if not callable(loop_fn):
        raise TypeError("loop_fn must be a callable")
    # Assume cell is a valid Keras layer/cell instance

    # Use provided or default values for loop controls
    parallel_iterations = parallel_iterations or 32 # Ensure default if None is passed

    with tf.name_scope(scope or "raw_rnn"):
        # Initialization call to loop_fn to get starting values
        time = tf.constant(0, dtype=dtypes.int32)
        # Pass None for cell_output, cell_state, loop_state initially
        (elements_finished, next_input, initial_state, emit_structure,
         init_loop_state) = loop_fn(time, None, None, None)

        # Flatten initial input to get batch size dynamically
        flat_input = nest.flatten(next_input)
        batch_size = tf.shape(flat_input[0])[0]
        # Try to get static batch size for TensorArray shape hints
        const_batch_size = tf.get_static_value(batch_size)

        # Validate initial state structure against cell's state_size
        nest.assert_same_structure(initial_state, cell.state_size, check_types=False)
        state = initial_state # Assume loop_fn provided a valid initial state

        # Ensure state tensors are correctly typed and placed
        flat_state = nest.flatten(state)
        flat_state = [ops.convert_to_tensor(s) for s in flat_state]
        state = nest.pack_sequence_as(structure=state, flat_sequence=flat_state)

        # Determine emit structure and prepare TensorArrays for outputs
        if emit_structure is None:
             raise ValueError("loop_fn must provide emit_structure")

        flat_emit_structure = nest.flatten(emit_structure)
        # Get element shapes for TensorArray definition
        flat_emit_element_shapes = [
            # Try concatenating static batch size with static element shape
            tensor_shape.TensorShape([const_batch_size]).concatenate(
                _maybe_tensor_shape_from_tensor(emit.shape) # Get element shape (excluding batch)
            )
            if const_batch_size is not None else None # Use None if batch size is dynamic
            for emit in flat_emit_structure
        ]
        flat_emit_dtypes = [emit.dtype for emit in flat_emit_structure]

        flat_emit_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i,
                size=0, # Dynamic size
                dynamic_size=True,
                element_shape=shape_i, # Pass None if dynamic batch size
                infer_shape=shape_i is None, # Infer shape if not fully defined
                name="rnn_output_%d" % i
            )
            for i, (dtype_i, shape_i) in enumerate(zip(flat_emit_dtypes, flat_emit_element_shapes))
        ]
        emit_ta = nest.pack_sequence_as(structure=emit_structure, flat_sequence=flat_emit_ta)

        # Prepare TensorArrays for storing states at each step
        flat_state_element_shapes = [
            tensor_shape.TensorShape([const_batch_size]).concatenate(
                _maybe_tensor_shape_from_tensor(s.shape[1:]) # Get element shape (excluding batch)
             ) if const_batch_size is not None else None
             for s in flat_state
        ]
        flat_state_dtypes = [s.dtype for s in flat_state]

        flat_state_ta = [
            tensor_array_ops.TensorArray(
                dtype=dtype_i,
                size=0, # Dynamic size
                dynamic_size=True,
                element_shape=shape_i,
                infer_shape=shape_i is None,
                name="rnn_state_%d" % i
            )
            for i, (dtype_i, shape_i) in enumerate(zip(flat_state_dtypes, flat_state_element_shapes))
        ]
        state_ta = nest.pack_sequence_as(structure=state, flat_sequence=flat_state_ta)

        # Create a zero value based on emit_structure for finished steps
        zero_emit = nest.map_structure(
            lambda emit: array_ops.zeros(
                # Use dynamic batch_size in shape specification
                array_ops.concat([tf.expand_dims(batch_size, 0), tf.shape(emit)[1:]], axis=0),
                dtype=emit.dtype
            ),
            emit_structure
        )

        # Handle optional loop state
        loop_state = init_loop_state if init_loop_state is not None else tf.constant(0, dtype=dtypes.int32)

        # --- tf.while_loop setup ---
        def condition(time, elements_finished, *_):
            # Continue loop if not all elements in the batch are finished
            return tf.logical_not(tf.reduce_all(elements_finished))

        def body(time, elements_finished, current_input, state_ta, emit_ta, current_state, loop_state):
            """The body of the tf.while_loop for the RNN."""
            # 1. Call the Keras cell instance
            # Keras cell expects state in a list, returns (outputs, new_states_list)
            # Pass constants explicitly to the cell's call method
            cell_output, next_state_list = cell(current_input, [current_state], constants=constants, training=False) # Assuming sampling is not training
            next_state = next_state_list[0] # Unpack the state tuple from the list

            # 2. Call the loop_fn to determine next inputs, finished status, etc.
            # loop_fn(time, cell_output, cell_state, loop_state)
            # It receives the raw cell output (e.g., h3) and the *next* full state tuple
            next_time = time + 1
            (next_finished, next_input, DONT_USE_STATE_FROM_LOOPFN, emit_output,
             next_loop_state) = loop_fn(next_time, cell_output, next_state, loop_state)

            # Use the state returned directly by the cell, not potentially modified by loop_fn
            final_next_state = next_state

            # Reuse previous loop_state if loop_fn returns None
            next_loop_state = loop_state if next_loop_state is None else next_loop_state

            # 3. Define _copy_some_through helper using tf.where
            def _copy_some_through(current, candidate):
                """Copies tensors using tf.where based on elements_finished mask."""
                def copy_fn(cur_i, cand_i):
                    if isinstance(cur_i, tensor_array_ops.TensorArray): return cand_i # Pass TAs through
                    # Broadcast finished mask [batch] to match tensor rank
                    try:
                        # Use tf.rank(cur_i) to handle potentially unknown ranks robustly
                        rank_i = tf.rank(cur_i)
                        elements_finished_expanded_shape = tf.concat([tf.shape(elements_finished), tf.ones([rank_i - 1], dtype=tf.int32)], axis=0)
                        elements_finished_expanded = tf.reshape(elements_finished, elements_finished_expanded_shape)
                        # Ensure candidate and current have compatible shapes for tf.where
                        # This might require broadcasting if shapes aren't identical beyond batch dim
                        return tf.where(elements_finished_expanded, cur_i, cand_i)
                    except Exception as e:
                        tf.print("Error in _copy_some_through reshape/where:", e)
                        tf.print("elements_finished shape:", tf.shape(elements_finished))
                        tf.print("cur_i shape:", tf.shape(cur_i))
                        tf.print("cand_i shape:", tf.shape(cand_i))
                        # Fallback or raise error
                        return cand_i # Or raise error
                return nest.map_structure(copy_fn, current, candidate)

            # 4. Apply tf.where to handle finished sequences
            emit_output = _copy_some_through(zero_emit, emit_output)
            final_next_state = _copy_some_through(current_state, final_next_state)

            # 5. Write results to TensorArrays
            emit_ta = nest.map_structure(lambda ta, emit: ta.write(time, emit), emit_ta, emit_output)
            state_ta = nest.map_structure(lambda ta, state: ta.write(time, state), state_ta, final_next_state)

            # 6. Update finished mask (logical OR with loop_fn's finished flag)
            elements_finished = tf.logical_or(elements_finished, next_finished)

            # 7. Return updated loop variables
            return (next_time, elements_finished, next_input, state_ta,
                    emit_ta, final_next_state, next_loop_state)

        # Initial loop variables for tf.while_loop
        loop_vars = [
            time, elements_finished, next_input, state_ta, emit_ta, state, loop_state
        ]

        # Define shape invariants for the loop
        # Need to explicitly handle the dynamic shape of 'phi' in the state
        initial_state_invariants = nest.map_structure(lambda s: s.get_shape(), initial_state)
        phi_invariant_shape = tf.TensorShape([state.phi.shape[0], None]) # Keep batch dim, allow char dim to be None
        state_invariants_list = list(initial_state_invariants)
        try:
            from rnn_cell import LSTMAttentionCellState # Import locally if needed
            phi_index = LSTMAttentionCellState._fields.index('phi')
            state_invariants_list[phi_index] = phi_invariant_shape
            final_state_invariants = LSTMAttentionCellState(*state_invariants_list)
        except (ImportError, ValueError, AttributeError):
             # Fallback if LSTMAttentionCellState import fails or structure changes
             if state_invariants_list: # Check if list is not empty
                 tf.print("Warning: Could not find 'phi' field by name in state. Setting last element shape invariant dynamically.")
                 state_invariants_list[-1] = phi_invariant_shape # Assume phi is last
             else:
                 tf.print("Warning: Cannot determine state invariants for LSTMAttentionCellState.")
             # Pack using the structure of the original state
             final_state_invariants = nest.pack_sequence_as(structure=initial_state, flat_sequence=state_invariants_list)


        emit_structure_invariants = nest.map_structure(lambda emit: emit.get_shape(), emit_structure)
        loop_vars_invariants = [
            tf.TensorShape([]),                  # time
            elements_finished.get_shape(),       # elements_finished
            nest.map_structure(lambda inp: inp.get_shape(), next_input), # next_input
            nest.map_structure(lambda ta: tf.TensorShape([]), state_ta), # state_ta (invariants handled by loop)
            nest.map_structure(lambda ta: tf.TensorShape([]), emit_ta),  # emit_ta (invariants handled by loop)
            final_state_invariants,              # state
            init_loop_state.get_shape() if init_loop_state is not None else tf.TensorShape([]) # loop_state
        ]

        # Execute the while_loop
        returned_loop_vars = tf.while_loop(
            condition, body, loop_vars=loop_vars,
            shape_invariants=loop_vars_invariants,   # Pass the defined invariants
            parallel_iterations=parallel_iterations, # Use arg/default
            swap_memory=swap_memory              # Use arg/default
        )

        # --- Process results ---
        final_state_ta = returned_loop_vars[3]
        final_emit_ta = returned_loop_vars[4]
        final_state = returned_loop_vars[5]
        # final_loop_state = returned_loop_vars[6] # If needed

        # Stack TensorArrays to get sequences
        output_tas = nest.flatten(final_emit_ta)
        outputs = [ta.stack() for ta in output_tas]
        state_tas = nest.flatten(final_state_ta)
        states = [ta.stack() for ta in state_tas]

        # Transpose from [time, batch, ...] to [batch, time, ...]
        # *** THE FIX: Use tf.range and tf.concat to build perm dynamically ***
        def build_perm(tensor):
            rank = tf.rank(tensor)
            # tf.range(start, limit, delta) -> generates [start, start+delta, ..., limit-delta]
            remaining_dims = tf.range(2, rank, dtype=tf.int32) # Generate 2, 3, ..., rank-1
            # Use tf.concat to build the permutation tensor: [1, 0] + [2, 3, ...]
            perm = tf.concat([tf.constant([1, 0], dtype=tf.int32), remaining_dims], axis=0)
            return perm

        outputs = [array_ops.transpose(o, perm=build_perm(o)) for o in outputs]
        states = [array_ops.transpose(s, perm=build_perm(s)) for s in states]
        # *** END FIX ***

        outputs = nest.pack_sequence_as(structure=emit_structure, flat_sequence=outputs)
        states = nest.pack_sequence_as(structure=state, flat_sequence=states)

        return (states, outputs, final_state)


# Pass parallel_iterations and swap_memory down if needed, otherwise use defaults
def rnn_free_run(cell, initial_state, sequence_length, bias, constants, initial_input=None,
                 parallel_iterations=32, swap_memory=False, # Add args here
                 scope='dynamic-rnn-free-run'):
    """
    TF2-compatible adaptation for free-running RNN using raw_rnn.
    Feeds back sampled outputs. Requires cell with output_function and (simplified) termination_condition.
    Args:
        cell: The Keras RNN cell instance (e.g., LSTMAttentionCell).
        initial_state: The starting state tuple for the cell.
        sequence_length: Maximum number of steps to run (scalar tensor or int).
        bias: Tensor of shape [batch_size] for sampling bias.
        constants: Tuple passed to the cell (e.g., attention values, lengths).
        initial_input: Optional first input point, shape [batch_size, 3]. If None, derived from cell.
        parallel_iterations: Passed to internal `raw_rnn` call.
        swap_memory: Passed to internal `raw_rnn` call.
        scope: Name scope for operations.
    Returns:
        Tuple: (all_states, all_outputs, final_state) from raw_rnn.
    """
    max_tsteps = sequence_length
    # Ensure initial_state is fully defined before trying to get batch_size
    flat_initial_state = nest.flatten(initial_state)
    if not flat_initial_state:
        raise ValueError("initial_state cannot be empty")
    batch_size = tf.shape(flat_initial_state[0])[0] # Get batch size from first element of flattened state

    with tf.name_scope(scope):
        # Determine the very first input (t=0)
        if initial_input is None:
            # Use cell.output_function with initial state's h3 and bias
            initial_h3 = initial_state.h3
            initial_input = cell.output_function(initial_h3, bias)
        # Ensure no gradients flow back through sampling process
        initial_input = tf.stop_gradient(initial_input)

        # Unpack constants needed by termination condition
        attention_values, attention_values_lengths = constants # Unpack constants tuple

        # Define the loop_fn for raw_rnn
        def loop_fn(time, cell_output, cell_state, loop_state):
            # `cell_output`: Raw output from cell (h3). None on first call.
            # `cell_state`: The full state tuple *after* the step (state at time `t`). None on first call.

            if cell_output is None: # First step (time == 0)
                # State is the initial_state passed to rnn_free_run
                next_cell_state = initial_state
                # Use the pre-calculated initial_input
                next_input = initial_input
                # On first step, assume not finished unless max_tsteps is 0 or negative
                elements_finished_time = tf.less_equal(max_tsteps, 0) # Finished if max_tsteps <= 0
                elements_finished_attn = tf.zeros([batch_size], dtype=tf.bool) # Cannot terminate based on attention yet
                elements_finished_eos = tf.zeros([batch_size], dtype=tf.bool) # Cannot terminate based on EOS yet
                # Emit the first input as the first output point
                emit_output = initial_input

            else: # Subsequent steps (time > 0)
                # State is the one computed by the cell for this time step
                next_cell_state = cell_state

                # Determine next input by sampling from the *current* state's output (h3)
                current_h3 = next_cell_state.h3
                next_input_candidate = cell.output_function(current_h3, bias) # Pass bias
                next_input_candidate = tf.stop_gradient(next_input_candidate)

                # Emit the sampled point for this time step
                emit_output = next_input_candidate # Shape [b, 3]

                # --- Termination Check ---
                # 1. Check attention window position using simplified cell method
                final_char_reached, past_final_char = cell.termination_condition(
                    next_cell_state, attention_values_lengths
                )
                # Stop if attention is strictly past the end character index
                elements_finished_attn = past_final_char

                # 2. Check if the *sampled output* indicates end-of-stroke (e=1)
                # emit_output has shape [b, 3] = [x, y, e]
                sampled_e = emit_output[:, 2] # Get the 'e' component, shape [b]
                is_eos_predicted = tf.equal(sampled_e, 1.0)

                # Combine attention and EOS condition: stop if (EOS predicted AND attention is at or past the end)
                elements_finished_eos = tf.logical_and(is_eos_predicted, final_char_reached)

                # 3. Check against max time steps (time is 0-based index of the *current* step being processed)
                # So, stop if time >= max_tsteps - 1 (meaning next step would be max_tsteps or greater)
                # Or simpler: stop if next_time (time + 1) >= max_tsteps
                elements_finished_time = tf.greater_equal(time + 1, max_tsteps)


                # Mask the next input based on *overall* finished status from *previous* step
                # (tf.while_loop uses finished status from end of previous iteration)
                # Note: elements_finished isn't available here, it's managed by raw_rnn body
                # The _copy_some_through in raw_rnn body handles zeroing output/state correctly.
                # So, we just need to provide the candidate next_input.
                next_input = next_input_candidate


            # Determine overall finished status for *this* step to be returned to raw_rnn
            # Stop if max time reached OR attention is past end OR (EOS predicted AND attention at/past end)
            elements_finished_step = tf.logical_or(
                elements_finished_time,
                tf.logical_or(elements_finished_attn, elements_finished_eos)
            )

            # Return values needed by raw_rnn body
            # finished, next_input, next_cell_state, emit_output, next_loop_state
            # Crucially, pass next_cell_state (the state *after* the RNN step) back to raw_rnn.
            # The raw_rnn's body will handle the tf.where logic based on the *previous* step's finished status.
            next_loop_state = None # Not using loop_state here
            return (elements_finished_step, next_input, next_cell_state, emit_output, next_loop_state)


        # Define the structure of the emitted output ([x, y, e]) using initial_input shape
        # Ensure initial_input is a tensor before accessing shape
        initial_input = tf.convert_to_tensor(initial_input, dtype_hint=tf.float32)
        emit_structure = tf.TensorSpec(shape=initial_input.shape[1:], dtype=initial_input.dtype)

        # Call raw_rnn with the sampling loop_fn
        all_states, all_outputs, final_state = raw_rnn(
            cell=cell,
            loop_fn=loop_fn,
            constants=constants, # Pass constants needed by cell.call
            parallel_iterations=parallel_iterations, # Pass arg down
            swap_memory=swap_memory,             # Pass arg down
            scope=scope
        )

        return all_states, all_outputs, final_state