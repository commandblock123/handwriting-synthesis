import tensorflow as tf

def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm_layer=None,
                dropout_rate=None, scope='dense-layer', training=False):
    """Applies a dense layer using tf.keras.layers.Dense."""
    layer = tf.keras.layers.Dense(
        units=output_units,
        activation=None,  # Apply activation after potential BN
        use_bias=bias,
        name=scope
    )
    z = layer(inputs)

    if batch_norm_layer is not None:
        z = batch_norm_layer(z, training=training)

    if activation is not None:
        z = activation(z)

    if dropout_rate is not None and dropout_rate > 0.0:
        dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        z = dropout_layer(z, training=training)

    return z


def time_distributed_dense_layer(
        inputs, output_units, bias=True, activation=None, batch_norm_layer=None,
        dropout_rate=None, scope='time-distributed-dense-layer', training=False):
    """Applies a shared dense layer to each timestep using TimeDistributed wrapper."""
    dense = tf.keras.layers.Dense(
        units=output_units,
        activation=None, # Apply activation after potential BN
        use_bias=bias
    )
    wrapper = tf.keras.layers.TimeDistributed(dense, name=scope)
    z = wrapper(inputs)

    # Apply BN/Dropout after TimeDistributed.
    # Note: If BN/Dropout were intended per time-step with shared weights in TF1 cell,
    # this implementation differs slightly. Usually applied after the sequence processing.
    if batch_norm_layer is not None:
        # Reshape for BN: (batch*time, features), apply BN, reshape back
        input_shape = tf.shape(z)
        reshaped_z = tf.reshape(z, [-1, output_units])
        bn_z = batch_norm_layer(reshaped_z, training=training)
        z = tf.reshape(bn_z, input_shape)

    if activation is not None:
        z = activation(z)

    if dropout_rate is not None and dropout_rate > 0.0:
        dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        z = dropout_layer(z, training=training)

    return z


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int (static shape)."""
    static_shape = tensor.shape
    if dim is None:
        return static_shape.as_list()
    else:
        # Handle negative dim indexing
        if dim < 0:
             dim = len(static_shape) + dim
        return static_shape[dim]


def rank(tensor):
    """Get tensor rank as python int (static shape)."""
    return tensor.shape.ndims
