from __future__ import print_function
import os
import time
import logging
from collections import deque
import pprint as pp
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import nest

# --- Enable XLA ---
tf.config.optimizer.set_jit(True)
print("XLA JIT Compilation Enabled:", tf.config.optimizer.get_jit())
# ------------------

import drawing
from data_frame import DataFrame
from rnn_cell import LSTMAttentionCell, LSTMAttentionCellState
from rnn_ops import rnn_free_run
from tf_utils import shape


tfd = tfp.distributions

# --- DataReader ---
class DataReader(object):
    def __init__(self, data_dir):
        data_cols = ['x', 'x_len', 'c', 'c_len']
        data = []
        for i in data_cols:
            filepath = os.path.join(data_dir, f'{i}.npy')
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}. Please ensure data is processed and in the correct directory.")
            data.append(np.load(filepath, allow_pickle=True))
        self.alphabet_size = len(drawing.alphabet)
        if isinstance(data[2], np.ndarray) and data[2].dtype == object:
             print("Padding character sequences ('c')...")
             max_c_len = max(len(arr) for arr in data[2]) if len(data[2]) > 0 else 0
             data[2] = np.array([np.pad(arr, (0, max_c_len - len(arr))) for arr in data[2]], dtype=np.int32)
             print(f"Padded 'c' to max length: {max_c_len}")
        elif isinstance(data[2], np.ndarray): print(f"'c' shape: {data[2].shape}")
        else: raise TypeError("Unsupported type for 'c' data")
        if isinstance(data[0], np.ndarray) and data[0].dtype == object:
             print("Padding stroke sequences ('x')...")
             max_x_len_data = max(len(arr) for arr in data[0]) if len(data[0]) > 0 else 0
             padded_x = []
             for arr in data[0]:
                  pad_width = ((0, max_x_len_data - len(arr)), (0, 0))
                  padded_x.append(np.pad(arr.astype(np.float32), pad_width, mode='constant', constant_values=0.0))
             data[0] = np.array(padded_x, dtype=np.float32)
             print(f"Padded 'x' to max length: {max_x_len_data}")
        elif isinstance(data[0], np.ndarray): print(f"'x' shape: {data[0].shape}")
        else: raise TypeError("Unsupported type for 'x' data")
        data[0] = data[0].astype(np.float32); data[1] = data[1].astype(np.int32); data[3] = data[3].astype(np.int32)
        self.test_df = DataFrame(columns=data_cols, data=data)
        self.train_df, self.val_df = self.test_df.train_test_split(train_size=0.95, random_state=2018)
        print('Train size:', len(self.train_df)); print('Validation size:', len(self.val_df)); print('Test size:', len(self.test_df))
    def train_batch_generator(self, batch_size): return self.batch_generator(batch_size=batch_size, df=self.train_df, shuffle=True, num_epochs=10000)
    def val_batch_generator(self, batch_size): return self.batch_generator(batch_size=batch_size, df=self.val_df, shuffle=True, num_epochs=10000)
    def test_batch_generator(self, batch_size): return self.batch_generator(batch_size=batch_size, df=self.test_df, shuffle=False, num_epochs=1, allow_smaller_final_batch=True)
    def batch_generator(self, batch_size, df, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        gen = df.batch_generator(batch_size=batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=allow_smaller_final_batch)
        for batch_df in gen:
            batch_data = {k: v.copy() for k, v in batch_df.dict.items()}
            batch_data['x_len'] = np.maximum(0, batch_data['x_len'] - 1).astype(np.int32)
            max_x_len_batch = np.max(batch_data['x_len']) if len(batch_data['x_len']) > 0 else 0
            max_c_len_batch = np.max(batch_data['c_len']) if len(batch_data['c_len']) > 0 else 0
            required_x_points = max_x_len_batch + 1
            if batch_data['x'].shape[1] < required_x_points:
                 pad_width = ((0, 0), (0, required_x_points - batch_data['x'].shape[1]), (0, 0)); batch_data['x'] = np.pad(batch_data['x'], pad_width, mode='constant', constant_values=0.0)
            input_x = batch_data['x'][:, :max_x_len_batch, :]; target_y = batch_data['x'][:, 1:max_x_len_batch + 1, :]; input_c = batch_data['c'][:, :max_c_len_batch]
            current_x_len = input_x.shape[1]; current_y_len = target_y.shape[1]; current_c_len = input_c.shape[1]
            if current_x_len < max_x_len_batch: input_x = np.pad(input_x, ((0, 0), (0, max_x_len_batch - current_x_len), (0, 0)), mode='constant', constant_values=0.0)
            if current_y_len < max_x_len_batch: target_y = np.pad(target_y, ((0, 0), (0, max_x_len_batch - current_y_len), (0, 0)), mode='constant', constant_values=0.0)
            if current_c_len < max_c_len_batch: input_c = np.pad(input_c, ((0, 0), (0, max_c_len_batch - current_c_len)), mode='constant', constant_values=0)
            yield {'x': tf.convert_to_tensor(input_x, dtype=tf.float32), 'y': tf.convert_to_tensor(target_y, dtype=tf.float32),
                   'x_len': tf.convert_to_tensor(batch_data['x_len'], dtype=tf.int32), 'c': tf.convert_to_tensor(input_c, dtype=tf.int32),
                   'c_len': tf.convert_to_tensor(batch_data['c_len'], dtype=tf.int32)}


# --- Keras Model Definition ---
class HandwritingRNN(tf.keras.Model):
    def __init__(self, lstm_size, output_mixture_components, attention_mixture_components, alphabet_size, name='handwriting_rnn', **kwargs):
        super(HandwritingRNN, self).__init__(name=name, **kwargs)
        self.lstm_size = lstm_size
        self.output_mixture_components = output_mixture_components
        self.attention_mixture_components = attention_mixture_components
        self.alphabet_size = alphabet_size
        self.output_units = output_mixture_components * 6 + 1

        # Create layers in __init__
        self.gmm_dense = tf.keras.layers.Dense(self.output_units, name='gmm_projection', dtype=tf.float32)
        self.attention_cell = LSTMAttentionCell(
            lstm_size=lstm_size, num_attn_mixture_components=attention_mixture_components,
            attention_values_dim=alphabet_size, num_output_mixture_components=output_mixture_components
        )
        self.attention_cell.set_gmm_layer(self.gmm_dense)
        # No self.rnn_layer wrapper needed

    # *** Reinstate Build Method ***
    def build(self, input_shape):
        """Explicitly build the attention cell's internal layers."""
        if self.built: return

        # --- Correctly Extract Shapes from input_shape (Dictionary of TensorSpecs) ---
        x_shape = input_shape['x'].shape # Get TensorShape from TensorSpec
        batch_size = x_shape[0] # Can be None
        time_steps = x_shape[1] # Can be None
        # c_shape = input_shape['c'].shape # [batch, char_time]

        # Determine input shapes for internal layers
        lstm1_input_dim = self.alphabet_size + 3
        lstm1_input_shape = tf.TensorShape([batch_size, lstm1_input_dim])

        lstm2_input_dim = 3 + self.lstm_size + self.alphabet_size
        lstm2_input_shape = tf.TensorShape([batch_size, lstm2_input_dim])

        lstm3_input_dim = 3 + self.lstm_size + self.alphabet_size
        lstm3_input_shape = tf.TensorShape([batch_size, lstm3_input_dim])

        attn_dense_input_dim = self.alphabet_size + 3 + self.lstm_size
        attn_dense_input_shape = tf.TensorShape([batch_size, attn_dense_input_dim])
        # --- End Shape Extraction ---

        # Build internal layers if they haven't been built
        if not self.attention_cell.lstm1.built: self.attention_cell.lstm1.build(lstm1_input_shape)
        if not self.attention_cell.lstm2.built: self.attention_cell.lstm2.build(lstm2_input_shape)
        if not self.attention_cell.lstm3.built: self.attention_cell.lstm3.build(lstm3_input_shape)
        if not self.attention_cell.attention_dense.built: self.attention_cell.attention_dense.build(attn_dense_input_shape)

        # Build the model's GMM dense layer
        gmm_input_shape = tf.TensorShape([batch_size, time_steps, self.lstm_size])
        if not self.gmm_dense.built: self.gmm_dense.build(gmm_input_shape)

        # Mark the main model as built
        super(HandwritingRNN, self).build(input_shape) # Pass the original dict signature
        logging.info(f"HandwritingRNN build completed for input signature: {input_shape}")


    def call(self, inputs, training=False):
        """Forward pass with manual RNN unrolling using tf.while_loop."""
        # (Call method remains unchanged from the previous version with the manual loop)
        x = inputs['x']
        c = inputs['c']
        x_len = inputs['x_len']
        c_len = inputs['c_len']

        batch_size = tf.shape(x)[0]
        max_time = tf.shape(x)[1]

        attention_values = tf.one_hot(c, depth=self.alphabet_size, dtype=tf.float32)
        constants = (attention_values, c_len)

        initial_state = self.attention_cell.get_initial_state(batch_size=batch_size, dtype=tf.float32)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time, dynamic_size=False,
                                     element_shape=tf.TensorShape([None, self.lstm_size]),
                                     clear_after_read=False, name='rnn_outputs_ta')

        time = tf.constant(0, dtype=tf.int32)
        initial_loop_vars = (time, initial_state, outputs_ta)

        def loop_condition(t, state, output_ta): return t < max_time
        def loop_body(t, current_state, current_outputs_ta):
            current_input = x[:, t, :]
            cell_output, new_state_list = self.attention_cell(current_input, states=[current_state], constants=constants, training=training)
            new_state = new_state_list[0]
            mask_t = t < x_len
            def _copy_some_through(current, candidate, condition):
                condition_reshaped = tf.reshape(condition, [-1, 1])
                return nest.map_structure(lambda cur_i, cand_i: tf.where(condition_reshaped, cand_i, cur_i), current, candidate)
            final_state_step = _copy_some_through(current_state, new_state, mask_t)
            final_output_step = tf.where(tf.reshape(mask_t, [-1, 1]), cell_output, tf.zeros_like(cell_output))
            current_outputs_ta = current_outputs_ta.write(t, final_output_step)
            return (t + 1, final_state_step, current_outputs_ta)

        initial_state_invariants = nest.map_structure(lambda s: s.get_shape(), initial_state)
        phi_invariant_shape = tf.TensorShape([initial_state.phi.shape[0], None])
        state_invariants_list = list(initial_state_invariants)
        try: phi_index = LSTMAttentionCellState._fields.index('phi'); state_invariants_list[phi_index] = phi_invariant_shape
        except (ValueError, AttributeError):
             if state_invariants_list: state_invariants_list[-1] = phi_invariant_shape
        final_state_invariants = LSTMAttentionCellState(*state_invariants_list)
        loop_vars_invariants = (tf.TensorShape([]), final_state_invariants, tf.TensorShape([]))

        _, final_state, outputs_ta = tf.while_loop(loop_condition, loop_body, loop_vars=initial_loop_vars, shape_invariants=loop_vars_invariants)

        rnn_outputs_stacked = outputs_ta.stack()
        rnn_outputs = tf.transpose(rnn_outputs_stacked, perm=[1, 0, 2])
        gmm_params = self.gmm_dense(rnn_outputs)
        return gmm_params, final_state


    # _parse_parameters method remains the same
    def _parse_parameters(self, z, bias=None, eps=1e-8, sigma_eps=1e-4):
        input_rank = tf.rank(z); num_components = self.output_mixture_components
        if bias is not None: bias_expanded = tf.reshape(bias, [-1] + [1]*(input_rank - 1))
        else: bias_expanded = 0.0
        pis_raw, sigmas_raw, rhos_raw, mus, es_raw = tf.split(z, [1*num_components, 2*num_components, 1*num_components, 2*num_components, 1], axis=-1)
        pis = tf.nn.softmax(pis_raw * (1.0 + bias_expanded), axis=-1); pis = tf.where(pis < .01, tf.zeros_like(pis), pis)
        sigmas = tf.clip_by_value(tf.exp(sigmas_raw - bias_expanded), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos_raw), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es_raw), eps, 1.0 - eps); es = tf.where(es < .01, tf.zeros_like(es), es)
        return pis, mus, sigmas, rhos, es

    # NLL_loss method remains the same
    def NLL_loss(self, y_true, gmm_params, sequence_lengths, eps=1e-8):
        pis, mus, sigmas, rhos, es = self._parse_parameters(gmm_params, bias=None)
        sigma_1, sigma_2 = tf.split(sigmas, 2, axis=-1); mu_1, mu_2 = tf.split(mus, 2, axis=-1); y_1, y_2, y_3 = tf.split(y_true, 3, axis=-1)
        sigma_1 = sigma_1 + eps; sigma_2 = sigma_2 + eps
        term1 = tf.square((y_1 - mu_1) / sigma_1); term2 = tf.square((y_2 - mu_2) / sigma_2); term3 = 2 * rhos * (y_1 - mu_1) * (y_2 - mu_2) / (sigma_1 * sigma_2)
        rho_squared = tf.square(rhos); z_denom = tf.maximum(1.0 - rho_squared, eps); Z = (term1 + term2 - term3) / z_denom
        log_norm_sqrt_term = 0.5 * tf.math.log(z_denom); log_norm = -tf.math.log(2.0 * np.pi) - tf.math.log(sigma_1) - tf.math.log(sigma_2) - log_norm_sqrt_term
        log_gaussian_likelihoods = -0.5 * Z + log_norm; log_pis = tf.math.log(tf.maximum(pis, eps)); gmm_log_likelihood = tf.reduce_logsumexp(log_pis + log_gaussian_likelihoods, axis=-1)
        es = tf.squeeze(es, axis=-1); y_3 = tf.squeeze(y_3, axis=-1); log_bernoulli_likelihood = tf.where(tf.equal(y_3, 1.0), tf.math.log(tf.maximum(es, eps)), tf.math.log(tf.maximum(1.0 - es, eps)))
        nll = -(gmm_log_likelihood + log_bernoulli_likelihood); maxlen_tensor = tf.shape(y_true)[1]; mask = tf.sequence_mask(sequence_lengths, maxlen=maxlen_tensor, dtype=tf.float32); masked_nll = nll * mask
        sequence_loss = tf.math.divide_no_nan(tf.reduce_sum(masked_nll, axis=1), tf.cast(sequence_lengths, tf.float32)); batch_loss = tf.reduce_mean(sequence_loss)
        return batch_loss

    # Sampling Method remains the same
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32), tf.TensorSpec(shape=[None], dtype=tf.int32),
        tf.TensorSpec(shape=[None], dtype=tf.float32), tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32, name="x_prime"),
        tf.TensorSpec(shape=[None], dtype=tf.int32, name="x_prime_len")])
    def sample(self, c, c_len, biases, max_tsteps, x_prime=None, x_prime_len=None):
        num_samples = tf.shape(c)[0]
        attention_values = tf.one_hot(c, depth=self.alphabet_size, dtype=tf.float32)
        constants = [attention_values, c_len]
        is_priming = tf.logical_and(x_prime is not None, x_prime_len is not None)
        def get_primed_state():
            prime_batch_size = tf.shape(x_prime)[0]; prime_max_time = tf.shape(x_prime)[1]
            prime_state = self.attention_cell.get_initial_state(batch_size=prime_batch_size, dtype=tf.float32)
            prime_time = tf.constant(0, dtype=tf.int32); prime_loop_vars = (prime_time, prime_state)
            prime_constants_tuple = (constants[0], constants[1])
            initial_prime_state_invariants = nest.map_structure(lambda s: s.get_shape(), prime_state)
            prime_phi_invariant_shape = tf.TensorShape([prime_state.phi.shape[0], None])
            prime_state_invariants_list = list(initial_prime_state_invariants)
            try: prime_phi_index = LSTMAttentionCellState._fields.index('phi'); prime_state_invariants_list[prime_phi_index] = prime_phi_invariant_shape
            except (ValueError, AttributeError):
                 if prime_state_invariants_list: prime_state_invariants_list[-1] = prime_phi_invariant_shape
            final_prime_state_invariants = LSTMAttentionCellState(*prime_state_invariants_list)
            prime_loop_vars_invariants = (tf.TensorShape([]), final_prime_state_invariants)
            def prime_cond(t, s): return t < prime_max_time
            def prime_body(t, current_s):
                 current_input = x_prime[:, t, :]
                 _, next_s_list = self.attention_cell(current_input, [current_s], constants=prime_constants_tuple, training=False)
                 mask_t = t < x_prime_len
                 def _copy_some_through_prime(current, candidate, condition):
                     condition_reshaped = tf.reshape(condition, [-1, 1])
                     return nest.map_structure(lambda cur_i, cand_i: tf.where(condition_reshaped, cand_i, cur_i), current, candidate )
                 final_s = _copy_some_through_prime(current_s, next_s_list[0], mask_t)
                 return (t + 1, final_s)
            _, final_prime_state = tf.while_loop(prime_cond, prime_body, prime_loop_vars, shape_invariants=prime_loop_vars_invariants)
            state = final_prime_state; first_input = self.attention_cell.output_function(state.h3, biases)
            return state, first_input
        def get_unprimed_state():
            state = self.attention_cell.get_initial_state(batch_size=num_samples, dtype=tf.float32)
            first_input = tf.concat([tf.zeros([num_samples, 2],dtype=tf.float32), tf.ones([num_samples, 1],dtype=tf.float32)], axis=1)
            return state, first_input
        initial_state, initial_input = tf.cond(is_priming, get_primed_state, get_unprimed_state)
        _, sampled_sequence, _ = rnn_free_run(
            cell=self.attention_cell, initial_state=initial_state, sequence_length=max_tsteps,
            bias=biases, constants=constants, initial_input=initial_input, scope='free_run_sampling')
        return sampled_sequence


# --- Training Setup & Loop ---

def init_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True); date_str = datetime.now().strftime('%Y-%m-%d_%H-%M'); log_file = f'log_{date_str}.txt'; log_path = os.path.join(log_dir, log_file)
    root_logger = logging.getLogger(); [root_logger.removeHandler(h) for h in root_logger.handlers[:]]
    logging.basicConfig(level=logging.INFO, format='[[%(asctime)s]] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    logging.info(f"Logging to: {log_path}")

def train(config):
    init_logging(config['log_dir'])
    logging.info(f"Starting training with config:\n{pp.pformat(config)}")
    try: dr = DataReader(data_dir=config['data_dir'])
    except Exception as e: logging.error(f"DataReader init failed: {e}", exc_info=True); return

    model = HandwritingRNN(
        lstm_size=config['lstm_size'], output_mixture_components=config['output_mixture_components'],
        attention_mixture_components=config['attention_mixture_components'], alphabet_size=len(drawing.alphabet)
    )
    logging.info("Model instantiated.")

    # --- Explicitly build the model ---
    try:
        logging.info("Building model explicitly with input signature...")
        example_batch = next(dr.val_batch_generator(batch_size=2))
        # Input signature for build excludes 'y'
        input_signature = { k: tf.TensorSpec(shape=v.shape, dtype=v.dtype)
                            for k, v in example_batch.items() if k != 'y' }
        model.build(input_signature)
    except Exception as e:
        logging.error(f"Failed to explicitly build model: {e}", exc_info=True); return
    # ------------------------------------

    # Optimizer setup
    boundaries = np.cumsum(config['patiences'])[:-1].tolist(); lr_values = config['learning_rates']; beta1_values = config['beta1_decays']
    learning_rate_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, lr_values)
    def create_optimizer(stage_index):
        lr = learning_rate_schedule; beta1 = beta1_values[stage_index]
        if config['optimizer'] == 'adam': return tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1)
        elif config['optimizer'] == 'rms': return tf.keras.optimizers.RMSprop(learning_rate=lr, rho=beta1, momentum=0.9)
        else: raise ValueError("Optimizer must be 'adam' or 'rms'")
    current_restart_idx = 0; optimizer = create_optimizer(current_restart_idx)

    # Checkpoint Management
    checkpoint_dir = config['checkpoint_dir']; os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(step=tf.Variable(0, dtype=tf.int64), optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # Restore checkpoint (model is already built)
    if config['warm_start_init_step'] > 0:
        restore_path = os.path.join(checkpoint_dir, f"ckpt-{config['warm_start_init_step']}")
        if os.path.exists(restore_path + ".index"):
            logging.info(f"Attempting restore from step: {restore_path}"); status = ckpt.restore(restore_path).expect_partial(); logging.info(f"Restored step {int(ckpt.step)}")
        else: logging.warning(f"Specific checkpoint {restore_path} not found.")
    elif ckpt_manager.latest_checkpoint:
        logging.info(f"Restoring from latest: {ckpt_manager.latest_checkpoint}"); status = ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial(); logging.info(f"Restored step {int(ckpt.step)}")
    else: logging.info("Initializing from scratch.")

    # Define training/validation steps
    @tf.function(reduce_retracing=True)
    def train_step(batch_data):
        with tf.GradientTape() as tape:
            gmm_params, _ = model(batch_data, training=True)
            loss = model.NLL_loss(batch_data['y'], gmm_params, batch_data['x_len'])
            if config['regularization_constant'] > 0.0: loss += config['regularization_constant'] * tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if 'bias' not in v.name])
        gradients = tape.gradient(loss, model.trainable_variables)
        valid_gradients = [(g, v) for g, v in zip(gradients, model.trainable_variables) if g is not None]
        all_finite = tf.reduce_all([tf.reduce_all(tf.math.is_finite(g)) for g, v in valid_gradients])
        def apply_gradients_fn():
            if config['grad_clip'] > 0:
                 grads_to_clip = [g for g, v in valid_gradients]; clipped_valid_grads, _ = tf.clip_by_global_norm(grads_to_clip, config['grad_clip'])
                 final_grads = []; idx=0
                 for g_orig, v in zip(gradients, model.trainable_variables):
                     if g_orig is not None: final_grads.append(clipped_valid_grads[idx]); idx+=1
                     else: final_grads.append(None)
                 grads_and_vars = list(zip(final_grads, model.trainable_variables))
            else: grads_and_vars = valid_gradients
            grads_to_apply = [(g, v) for g, v in grads_and_vars if g is not None]
            if grads_to_apply: optimizer.apply_gradients(grads_to_apply)
            return loss
        def skip_update_fn(): tf.print("Warning: NaN/Inf grads step", ckpt.step, ". Skipping update."); return loss
        final_loss = tf.cond(all_finite, apply_gradients_fn, skip_update_fn)
        return final_loss

    @tf.function(reduce_retracing=True)
    def validation_step(batch_data):
        gmm_params, _ = model(batch_data, training=False)
        loss = model.NLL_loss(batch_data['y'], gmm_params, batch_data['x_len'])
        return loss

    # --- Training Loop ---
    train_loss_history = deque(maxlen=config['loss_averaging_window']); val_loss_history = deque(maxlen=config['loss_averaging_window'])
    train_time_history = deque(maxlen=config['loss_averaging_window']); val_time_history = deque(maxlen=config['loss_averaging_window'])
    initial_step = int(ckpt.step); best_validation_loss = float('inf')

    # Calculate initial val loss (model is already built)
    try:
        logging.info("Calculating initial validation loss...")
        val_generator_init = dr.val_batch_generator(config['validation_batch_size']); initial_val_losses = []
        for _ in range(min(5, len(dr.val_df) // config['validation_batch_size'] + 1)):
            val_batch = next(val_generator_init); loss_val = validation_step(val_batch).numpy()
            if not (np.isnan(loss_val) or np.isinf(loss_val)): initial_val_losses.append(loss_val)
        if initial_val_losses: best_validation_loss = np.mean(initial_val_losses)
        logging.info(f"Initial Best Validation Loss set to: {best_validation_loss:.8f}")
        del val_generator_init
    except Exception as e: logging.warning(f"Could not calculate initial validation loss: {e}")

    steps_since_best_val = 0; current_patience = config['patiences'][current_restart_idx]
    current_batch_size = config['batch_sizes'][current_restart_idx]
    train_generator = dr.train_batch_generator(current_batch_size)
    val_generator = dr.val_batch_generator(config['validation_batch_size'])
    logging.info(f"Starting training loop from step {initial_step}")

    for step in range(initial_step, config['num_training_steps']):
        ckpt.step.assign_add(1); current_step = int(ckpt.step)
        if current_step % config['log_interval'] == 0:
            try:
                 val_start = time.time(); val_batch = next(val_generator); val_loss = validation_step(val_batch); val_loss_val = val_loss.numpy()
                 if np.isnan(val_loss_val) or np.isinf(val_loss_val): logging.warning(f"Val loss NaN/Inf step {current_step}")
                 else: val_loss_history.append(val_loss_val)
                 val_time_history.append(time.time() - val_start)
            except StopIteration: logging.warning("Val generator exhausted"); val_generator = dr.val_batch_generator(config['validation_batch_size'])
            except Exception as e: logging.error(f"Val step {current_step} error: {e}", exc_info=True)
        try:
             train_start = time.time(); train_batch = next(train_generator); train_loss = train_step(train_batch); train_loss_val = train_loss.numpy()
             if np.isnan(train_loss_val) or np.isinf(train_loss_val): logging.warning(f"Train loss NaN/Inf step {current_step}")
             else: train_loss_history.append(train_loss_val)
             train_time_history.append(time.time() - train_start)
        except StopIteration: logging.warning("Train generator exhausted"); train_generator = dr.train_batch_generator(current_batch_size); continue
        except Exception as e: logging.error(f"Train step {current_step} error: {e}", exc_info=True); continue
        if current_step % config['log_interval'] == 0 and current_step > initial_step:
            avg_train_loss = np.mean([l for l in train_loss_history if not np.isnan(l)]) if train_loss_history else np.nan; avg_val_loss = np.mean([l for l in val_loss_history if not np.isnan(l)]) if val_loss_history else np.nan
            avg_train_time = np.mean(train_time_history) if train_time_history else np.nan; avg_val_time = np.mean(val_time_history) if val_time_history else np.nan
            current_lr = optimizer.learning_rate.numpy()
            log_msg = (f"[[step {current_step:>8}]] [[train {avg_train_time:>4.2f}s]] loss: {avg_train_loss:<12.8f} [[val {avg_val_time:>4.2f}s]] loss: {avg_val_loss:<12.8f} (LR: {current_lr:.1e})")
            logging.info(log_msg); current_avg_val_loss = avg_val_loss
            if not np.isnan(current_avg_val_loss) and current_avg_val_loss < best_validation_loss:
                best_validation_loss = current_avg_val_loss; steps_since_best_val = 0
                if current_step >= config['min_steps_to_checkpoint']: save_path = ckpt_manager.save(); logging.info(f"Val loss improved. Saved: {save_path}")
            elif not np.isnan(current_avg_val_loss): steps_since_best_val += config['log_interval']
            if steps_since_best_val >= current_patience:
                logging.info(f"Patience ({current_patience} steps) exceeded.")
                if current_restart_idx < len(config['learning_rates']) - 1:
                    logging.info(f"--- Attempting restart {current_restart_idx + 1} ---"); best_ckpt = ckpt_manager.latest_checkpoint
                    if best_ckpt:
                         logging.info(f"Restoring best: {best_ckpt}")
                         try: ckpt.restore(best_ckpt).expect_partial(); logging.info(f"Restored step {int(ckpt.step)}."); steps_since_best_val = 0; train_loss_history.clear(); val_loss_history.clear(); train_time_history.clear(); val_time_history.clear()
                         except Exception as restore_e: logging.error(f"Restore failed: {restore_e}. Stopping."); break
                    else: logging.warning("No checkpoint to restore. Stopping."); break
                    current_restart_idx += 1; logging.info(f"--- Starting stage {current_restart_idx + 1} ---")
                    current_patience = config['patiences'][current_restart_idx]; new_batch_size = config['batch_sizes'][current_restart_idx]
                    if beta1_values[current_restart_idx] != beta1_values[current_restart_idx-1]: logging.info("Re-creating optimizer."); optimizer = create_optimizer(current_restart_idx); ckpt.optimizer = optimizer
                    if new_batch_size != current_batch_size: logging.info(f"Updating batch size to {new_batch_size}"); current_batch_size = new_batch_size; train_generator = dr.train_batch_generator(current_batch_size)
                    logging.info(f"New patience: {current_patience}.")
                else: logging.info("No more restarts. Stopping."); break

    logging.info("Training finished.")
    if int(ckpt.step) > initial_step: final_save_path = ckpt_manager.save(); logging.info(f"Saved final checkpoint after {int(ckpt.step)} steps: {final_save_path}")

if __name__ == '__main__':
    train_config = {
        'data_dir': 'data/processed/', 'log_dir': 'logs_tf2', 'checkpoint_dir': 'checkpoints_tf2',
        'prediction_dir': 'predictions_tf2', 'learning_rates': [0.0001, 5e-05, 2e-05],
        'beta1_decays': [0.9, 0.9, 0.9], 'batch_sizes': [32, 64, 64], 'patiences': [2000, 6000-1500, 5000],
        'optimizer': 'rms', 'num_training_steps': 100000, 'warm_start_init_step': 0,
        'regularization_constant': 0.0, 'grad_clip': 10.0, 'lstm_size': 400,
        'output_mixture_components': 20, 'attention_mixture_components': 10,
        'min_steps_to_checkpoint': 85, 'log_interval': 30, 'loss_averaging_window': 100,
        'validation_batch_size': 64,
    }
    os.makedirs(train_config['log_dir'], exist_ok=True); os.makedirs(train_config['checkpoint_dir'], exist_ok=True); os.makedirs(train_config['prediction_dir'], exist_ok=True)
    train(train_config)