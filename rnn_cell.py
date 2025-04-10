from collections import namedtuple
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tf_utils import shape

tfd = tfp.distributions

# Keep the named tuple for state clarity
LSTMAttentionCellState = namedtuple(
    'LSTMAttentionCellState',
    ['h1', 'c1', 'h2', 'c2', 'h3', 'c3', 'alpha', 'beta', 'kappa', 'w', 'phi']
)

class LSTMAttentionCell(tf.keras.layers.Layer):
    """
    LSTM Cell with Attention mechanism adapted for TF2 Keras Layers.
    Includes internal methods for sampling support.
    """

    def __init__(
        self,
        lstm_size,
        num_attn_mixture_components,
        attention_values_dim, # Dimension of one-hot encoded alphabet
        num_output_mixture_components,
        name='lstm_attention_cell',
        **kwargs
    ):
        # Ensure the base Layer's __init__ is called with dtype=float32 by default if needed
        if 'dtype' not in kwargs:
            kwargs['dtype'] = tf.float32
        super(LSTMAttentionCell, self).__init__(name=name, **kwargs)

        self.lstm_size = lstm_size
        self.num_attn_mixture_components = num_attn_mixture_components
        self.attention_values_dim = attention_values_dim
        self.num_output_mixture_components = num_output_mixture_components
        self.output_units = 6 * self.num_output_mixture_components + 1 # For GMM projection

        # Define LSTM Cells, explicitly setting dtype
        self.lstm1 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm1', dtype=self.dtype)
        self.lstm2 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm2', dtype=self.dtype)
        self.lstm3 = tf.keras.layers.LSTMCell(self.lstm_size, name='lstm3', dtype=self.dtype)

        # Dense layer for attention parameters (alpha, beta, kappa)
        self.attention_dense = tf.keras.layers.Dense(
            units=3 * self.num_attn_mixture_components,
            activation=tf.nn.softplus, # Apply softplus here as in original
            name='attention_dense',
            dtype=self.dtype
        )

        # Placeholder for the GMM projection layer (set externally)
        self.gmm_dense_layer = None

        # Define state size structure using the named tuple
        self._state_size_tuple = LSTMAttentionCellState(
            tf.TensorShape([self.lstm_size]), tf.TensorShape([self.lstm_size]), # h1, c1
            tf.TensorShape([self.lstm_size]), tf.TensorShape([self.lstm_size]), # h2, c2
            tf.TensorShape([self.lstm_size]), tf.TensorShape([self.lstm_size]), # h3, c3
            tf.TensorShape([self.num_attn_mixture_components]), # alpha
            tf.TensorShape([self.num_attn_mixture_components]), # beta
            tf.TensorShape([self.num_attn_mixture_components]), # kappa
            tf.TensorShape([self.attention_values_dim]), # w
            tf.TensorShape([None]) # phi (dynamic length)
        )

    @property
    def state_size(self):
        """Required property for Keras RNN layer (though we bypass RNN layer in call)."""
        return self._state_size_tuple

    @property
    def output_size(self):
        """Required property for Keras RNN layer (defines cell output)."""
        return self.lstm_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Returns the initial state tuple for the cell."""
        # Use layer's dtype if not provided, matching layer's compute/variable types
        if dtype is None:
            dtype = self.dtype

        if batch_size is None:
             if inputs is not None:
                 batch_size = tf.shape(inputs)[0]
             else:
                 raise ValueError("batch_size must be provided if inputs is None")

        # *** CORRECTED Call to LSTMCell.get_initial_state: Pass only batch_size ***
        # The dtype is inferred from the LSTMCell's own dtype property set during its __init__
        h1, c1 = self.lstm1.get_initial_state(batch_size=batch_size)
        h2, c2 = self.lstm2.get_initial_state(batch_size=batch_size)
        h3, c3 = self.lstm3.get_initial_state(batch_size=batch_size)
        # *** END CORRECTION ***

        # Initialize other state components to zeros using the determined dtype
        zero_attn_params = tf.zeros([batch_size, self.num_attn_mixture_components], dtype=dtype)
        zero_w = tf.zeros([batch_size, self.attention_values_dim], dtype=dtype)
        zero_phi = tf.zeros([batch_size, 1], dtype=dtype) # Placeholder shape

        return LSTMAttentionCellState(
            h1=h1, c1=c1, h2=h2, c2=c2, h3=h3, c3=c3,
            alpha=zero_attn_params, beta=zero_attn_params, kappa=zero_attn_params,
            w=zero_w, phi=zero_phi
        )

    def set_gmm_layer(self, layer):
        """Sets the external GMM projection layer needed for sampling helpers."""
        if not isinstance(layer, tf.keras.layers.Dense):
             raise TypeError("GMM layer must be a tf.keras.layers.Dense instance")
        self.gmm_dense_layer = layer

    def call(self, inputs, states, constants, training=False):
        """ Performs one step of the LSTMAttentionCell computation. """
        prev_state = states[0] # Keras RNN wraps state in a list / Manual loop passes tuple directly
        # Ensure constants is unpacked correctly depending on how it's passed
        if isinstance(constants, (list, tuple)) and len(constants) == 2:
            attention_values, attention_values_lengths = constants
        else:
            # Handle unexpected constants format if necessary
            raise ValueError("constants must be a list/tuple of (attention_values, attention_values_lengths)")

        batch_size = tf.shape(inputs)[0]
        char_len = tf.shape(attention_values)[1]

        # LSTM 1
        s1_in = tf.concat([prev_state.w, inputs], axis=1)
        s1_out, s1_state_list = self.lstm1(s1_in, states=[prev_state.h1, prev_state.c1], training=training)
        s1_h, s1_c = s1_state_list[0], s1_state_list[1]

        # Attention Mechanism
        attention_inputs = tf.concat([prev_state.w, inputs, s1_out], axis=1)
        attention_params = self.attention_dense(attention_inputs)
        alpha, beta, kappa_delta = tf.split(attention_params, 3, axis=1)
        kappa = prev_state.kappa + kappa_delta / 25.0
        beta = tf.clip_by_value(beta, .01, np.inf)
        kappa_exp = tf.expand_dims(kappa, 2); alpha_exp = tf.expand_dims(alpha, 2); beta_exp = tf.expand_dims(beta, 2)
        u = tf.cast(tf.range(char_len), dtype=tf.float32); u = tf.reshape(u, (1, 1, char_len))
        phi_components = alpha_exp * tf.exp(-tf.square(kappa_exp - u) / beta_exp)
        phi = tf.reduce_sum(phi_components, axis=1)
        sequence_mask = tf.sequence_mask(attention_values_lengths, maxlen=char_len, dtype=tf.float32)
        phi = phi * sequence_mask
        phi_sum = tf.reduce_sum(phi, axis=1, keepdims=True)
        phi = tf.math.divide_no_nan(phi, phi_sum)
        w = tf.einsum('bc,bcd->bd', phi, attention_values)

        # LSTM 2
        s2_in = tf.concat([inputs, s1_out, w], axis=1)
        s2_out, s2_state_list = self.lstm2(s2_in, states=[prev_state.h2, prev_state.c2], training=training)
        s2_h, s2_c = s2_state_list[0], s2_state_list[1]

        # LSTM 3
        s3_in = tf.concat([inputs, s2_out, w], axis=1)
        s3_out, s3_state_list = self.lstm3(s3_in, states=[prev_state.h3, prev_state.c3], training=training)
        s3_h, s3_c = s3_state_list[0], s3_state_list[1]

        new_state = LSTMAttentionCellState(
            h1=s1_h, c1=s1_c, h2=s2_h, c2=s2_c, h3=s3_h, c3=s3_c,
            alpha=alpha, beta=beta, kappa=kappa, w=w, phi=phi)

        # Return output (h3) and new state (wrapped in list)
        return s3_out, [new_state]


    # _parse_gmm_params method remains the same
    def _parse_gmm_params(self, gmm_params, bias, eps=1e-8, sigma_eps=1e-4):
        if self.gmm_dense_layer is None: raise ValueError("GMM dense layer has not been set.")
        bias_expanded = tf.expand_dims(bias, 1)
        pis_raw, sigmas_raw, rhos_raw, mus, es_raw = tf.split(gmm_params, [
                1*self.num_output_mixture_components, 2*self.num_output_mixture_components,
                1*self.num_output_mixture_components, 2*self.num_output_mixture_components, 1], axis=-1)
        pis = tf.nn.softmax(pis_raw * (1.0 + bias_expanded), axis=-1)
        pis = tf.where(pis < .01, tf.zeros_like(pis), pis)
        sigmas = tf.clip_by_value(tf.exp(sigmas_raw - bias_expanded), sigma_eps, np.inf)
        rhos = tf.clip_by_value(tf.tanh(rhos_raw), eps - 1.0, 1.0 - eps)
        es = tf.clip_by_value(tf.nn.sigmoid(es_raw), eps, 1.0 - eps)
        es = tf.where(es < .01, tf.zeros_like(es), es)
        return pis, mus, sigmas, rhos, es

    # output_function method remains the same
    def output_function(self, final_lstm_output, bias):
        if self.gmm_dense_layer is None: raise ValueError("GMM layer not set.")
        gmm_params = self.gmm_dense_layer(final_lstm_output)
        pis, mus, sigmas, rhos, es = self._parse_gmm_params(gmm_params, bias)
        batch_size = tf.shape(final_lstm_output)[0]
        sigma1, sigma2 = tf.split(sigmas, 2, axis=1); mu1, mu2 = tf.split(mus, 2, axis=1)
        components = []
        for i in range(self.num_output_mixture_components):
            loc_i = tf.stack([mu1[:, i], mu2[:, i]], axis=1)
            cov_i = tf.stack([tf.square(sigma1[:, i]), rhos[:, i]*sigma1[:, i]*sigma2[:, i],
                              rhos[:, i]*sigma1[:, i]*sigma2[:, i], tf.square(sigma2[:, i])], axis=1)
            cov_i = tf.reshape(cov_i, [batch_size, 2, 2])
            try: scale_tril_i = tf.linalg.cholesky(cov_i); mvn_i = tfd.MultivariateNormalTriL(loc=loc_i, scale_tril=scale_tril_i)
            except tf.errors.InvalidArgumentError: mvn_i = tfd.MultivariateNormalFullCovariance(loc=loc_i, covariance_matrix=cov_i)
            components.append(mvn_i)
        locs_stacked = tf.stack([comp.loc for comp in components], axis=1)
        covs_stacked = tf.stack([comp.covariance() for comp in components], axis=1)
        batched_mvn_components = tfd.MultivariateNormalFullCovariance(loc=locs_stacked, covariance_matrix=covs_stacked)
        cat = tfd.Categorical(probs=pis)
        mixture_dist = tfd.MixtureSameFamily(mixture_distribution=cat, components_distribution=batched_mvn_components)
        bern_dist = tfd.Bernoulli(probs=es, dtype=tf.float32)
        sampled_coords = mixture_dist.sample(); sampled_e = bern_dist.sample()
        return tf.concat([sampled_coords, sampled_e], axis=1)

    # termination_condition method remains the same
    def termination_condition(self, state, attention_values_lengths):
        char_idx = tf.cast(tf.argmax(state.phi, axis=1), dtype=tf.int32)
        final_char_reached = char_idx >= (attention_values_lengths - 1)
        past_final_char = char_idx >= attention_values_lengths
        return final_char_reached, past_final_char