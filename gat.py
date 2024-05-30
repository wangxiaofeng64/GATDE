import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
import warnings
from functools import wraps
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras import constraints, initializers, regularizers


def transpose(a, perm=None, name=None):

    if K.is_sparse(a):
        transpose_op = tf.sparse.transpose
    else:
        transpose_op = tf.transpose

    if perm is None:
        perm = (1, 0)  # Make explicit so that shape will always be preserved
    return transpose_op(a, perm=perm, name=name)


def reshape(a, shape=None, name=None):

    if K.is_sparse(a):
        reshape_op = tf.sparse.reshape
    else:
        reshape_op = tf.reshape

    return reshape_op(a, shape=shape, name=name)


def repeat(x, repeats):

    x = tf.expand_dims(x, 1)
    max_repeats = tf.reduce_max(repeats)
    tile_repeats = [1, max_repeats]
    arr_tiled = tf.tile(x, tile_repeats)
    mask = tf.less(tf.range(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result


def segment_top_k(x, I, ratio):

    rt = tf.RaggedTensor.from_value_rowids(x, I)
    row_lengths = rt.row_lengths()
    dense = rt.to_tensor(default_value=-np.inf)
    indices = tf.cast(tf.argsort(dense, direction="DESCENDING"), tf.int64)
    row_starts = tf.cast(rt.row_starts(), tf.int64)
    indices = indices + tf.expand_dims(row_starts, 1)
    row_lengths = tf.cast(
        tf.math.ceil(ratio * tf.cast(row_lengths, tf.float32)), tf.int32
    )
    return tf.RaggedTensor.from_tensor(indices, row_lengths).values


def indices_to_mask(indices, shape, dtype=tf.bool):

    indices = tf.convert_to_tensor(indices, dtype_hint=tf.int64)
    if indices.shape.ndims == 1:
        assert isinstance(shape, int) or shape.shape.ndims == 0
        indices = tf.expand_dims(indices, axis=1)
        if isinstance(shape, int):
            shape = tf.TensorShape([shape])
        else:
            shape = tf.expand_dims(shape, axis=0)
    else:
        indices.shape.assert_has_rank(2)
    assert indices.dtype.is_integer
    nnz = tf.shape(indices)[0]
    indices = tf.cast(indices, tf.int64)
    shape = tf.cast(shape, tf.int64)
    return tf.scatter_nd(indices, tf.ones((nnz,), dtype=dtype), shape)


SINGLE = 1  # Single mode    rank(x) = 2, rank(a) = 2
DISJOINT = SINGLE  # Disjoint mode  rank(x) = 2, rank(a) = 2
BATCH = 3  # Batch mode     rank(x) = 3, rank(a) = 3
MIXED = 4  # Mixed mode     rank(x) = 3, rank(a) = 2


def disjoint_signal_to_batch(X, I):

    I = tf.cast(I, tf.int32)
    num_nodes = tf.math.segment_sum(tf.ones_like(I), I)
    start_index = tf.cumsum(num_nodes, exclusive=True)
    n_graphs = tf.shape(num_nodes)[0]
    max_n_nodes = tf.reduce_max(num_nodes)
    batch_n_nodes = tf.shape(I)[0]
    feature_dim = tf.shape(X)[-1]

    index = tf.range(batch_n_nodes)
    index = (index - tf.gather(start_index, I)) + (I * max_n_nodes)
    dense = tf.zeros((n_graphs * max_n_nodes, feature_dim), dtype=X.dtype)
    dense = tf.tensor_scatter_nd_update(dense, index[..., None], X)

    batch = tf.reshape(dense, (n_graphs, max_n_nodes, feature_dim))

    return batch


def disjoint_adjacency_to_batch(A, I):

    I = tf.cast(I, tf.int64)
    indices = A.indices
    values = A.values
    i_nodes, j_nodes = indices[:, 0], indices[:, 1]

    graph_sizes = tf.math.segment_sum(tf.ones_like(I), I)
    max_n_nodes = tf.reduce_max(graph_sizes)
    n_graphs = tf.shape(graph_sizes)[0]

    offset = tf.gather(I, i_nodes)
    offset = tf.gather(tf.cumsum(graph_sizes, exclusive=True), offset)

    relative_j_nodes = j_nodes - offset
    relative_i_nodes = i_nodes - offset

    spaced_i_nodes = tf.gather(I, i_nodes) * max_n_nodes + relative_i_nodes
    new_indices = tf.transpose(tf.stack([spaced_i_nodes, relative_j_nodes]))

    n_graphs = tf.cast(n_graphs, new_indices.dtype)
    max_n_nodes = tf.cast(max_n_nodes, new_indices.dtype)

    dense_adjacency = tf.scatter_nd(
        new_indices, values, (n_graphs * max_n_nodes, max_n_nodes)
    )
    batch = tf.reshape(dense_adjacency, (n_graphs, max_n_nodes, max_n_nodes))

    return batch


def autodetect_mode(x, a):

    x_ndim = K.ndim(x)
    a_ndim = K.ndim(a)
    if x_ndim == 2 and a_ndim == 2:
        return SINGLE
    elif x_ndim == 3 and a_ndim == 3:
        return BATCH
    elif x_ndim == 3 and a_ndim == 2:
        return MIXED
    else:
        raise ValueError(
            "Unknown mode for inputs x, a with ranks {} and {}"
            "respectively.".format(x_ndim, a_ndim)
        )

from tensorflow.keras import activations, constraints, initializers, regularizers

LAYER_KWARGS = {"activation", "use_bias"}
KERAS_KWARGS = {
    "trainable",
    "name",
    "dtype",
    "dynamic",
    "input_dim",
    "input_shape",
    "batch_input_shape",
    "batch_size",
    "weights",
    "activity_regularizer",
    "autocast",
    "implementation",
}


def is_layer_kwarg(key):
    return key not in KERAS_KWARGS and (
        key.endswith("_initializer")
        or key.endswith("_regularizer")
        or key.endswith("_constraint")
        or key in LAYER_KWARGS
    )


def is_keras_kwarg(key):
    return key in KERAS_KWARGS


def deserialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.get(attr)
    if key.endswith("_regularizer"):
        return regularizers.get(attr)
    if key.endswith("_constraint"):
        return constraints.get(attr)
    if key == "activation":
        return activations.get(attr)
    return attr


def serialize_kwarg(key, attr):
    if key.endswith("_initializer"):
        return initializers.serialize(attr)
    if key.endswith("_regularizer"):
        return regularizers.serialize(attr)
    if key.endswith("_constraint"):
        return constraints.serialize(attr)
    if key == "activation":
        return activations.serialize(attr)
    if key == "use_bias":
        return attr

import warnings
from functools import wraps

import tensorflow as tf
from tensorflow.keras.layers import Layer



class Conv(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if is_keras_kwarg(k)})
        self.supports_masking = True
        self.kwargs_keys = []
        for key in kwargs:
            if is_layer_kwarg(key):
                attr = kwargs[key]
                attr = deserialize_kwarg(key, attr)
                self.kwargs_keys.append(key)
                setattr(self, key, attr)
        self.call = check_dtypes_decorator(self.call)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        base_config = super().get_config()
        keras_config = {}
        for key in self.kwargs_keys:
            keras_config[key] = serialize_kwarg(key, getattr(self, key))
        return {**base_config, **keras_config, **self.config}

    @property
    def config(self):
        return {}

    @staticmethod
    def preprocess(a):
        return a


def check_dtypes_decorator(call):
    @wraps(call)
    def _inner_check_dtypes(inputs, **kwargs):
        inputs = check_dtypes(inputs)
        return call(inputs, **kwargs)

    return _inner_check_dtypes


def check_dtypes(inputs):
    for value in inputs:
        if not hasattr(value, "dtype"):
            # It's not a valid tensor.
            return inputs

    if len(inputs) == 2:
        x, a = inputs
        e = None
    elif len(inputs) == 3:
        x, a, e = inputs
    else:
        return inputs

    if a.dtype in (tf.int32, tf.int64) and x.dtype in (
        tf.float16,
        tf.float32,
        tf.float64,
    ):
        warnings.warn(
            f"The adjacency matrix of dtype {a.dtype} is incompatible with the dtype "
            f"of the node features {x.dtype} and has been automatically cast to "
            f"{x.dtype}."
        )
        a = tf.cast(a, x.dtype)

    output = [_ for _ in [x, a, e] if _ is not None]
    return output
   
class GATConv(Conv):

    def __init__(
        self,
        channels,
        attn_heads=1,
        concat_heads=True,
        dropout_rate=0.5,
        return_attn_coef=True,
        add_self_loops=True,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        attn_kernel_initializer="glorot_uniform",
        kernel_regularizer=None,
        bias_regularizer=None,
        attn_kernel_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        attn_kernel_constraint=None,
        **kwargs,
    ):
        super().__init__(
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.channels = channels
        self.attn_heads = attn_heads
        self.concat_heads = concat_heads
        self.dropout_rate = dropout_rate
        self.return_attn_coef = return_attn_coef
        self.add_self_loops = add_self_loops
        self.attn_kernel_initializer = initializers.get(attn_kernel_initializer)
        self.attn_kernel_regularizer = regularizers.get(attn_kernel_regularizer)
        self.attn_kernel_constraint = constraints.get(attn_kernel_constraint)

        if concat_heads:
            self.output_dim = self.channels * self.attn_heads
        else:
            self.output_dim = self.channels

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            name="kernel",
            shape=[input_dim, self.attn_heads, self.channels],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
        )
        self.attn_kernel_self = self.add_weight(
            name="attn_kernel_self",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        self.attn_kernel_neighs = self.add_weight(
            name="attn_kernel_neigh",
            shape=[self.channels, self.attn_heads, 1],
            initializer=self.attn_kernel_initializer,
            regularizer=self.attn_kernel_regularizer,
            constraint=self.attn_kernel_constraint,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=[self.output_dim],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name="bias",
            )

        self.dropout = Dropout(self.dropout_rate, dtype=self.dtype)
        self.built = True

    def call(self, inputs, mask=None):
        x, a = inputs

        mode = autodetect_mode(x, a)
        if mode == SINGLE and K.is_sparse(a):
            output, attn_coef = self._call_single(x, a)
        else:
            if K.is_sparse(a):
                a = tf.sparse.to_dense(a)
            output, attn_coef = self._call_dense(x, a)

        if self.concat_heads:
            shape = tf.concat(
                (tf.shape(output)[:-2], [self.attn_heads * self.channels]), axis=0
            )
            output = tf.reshape(output, shape)
        else:
            output = tf.reduce_mean(output, axis=-2)

        if self.use_bias:
            output += self.bias
        if mask is not None:
            output *= mask[0]
        output = self.activation(output)

        if self.return_attn_coef:
            return output, attn_coef
        else:
            return output

    def _call_single(self, x, a):
        # Reshape kernels for efficient message-passing
        kernel = tf.reshape(self.kernel, (-1, self.attn_heads * self.channels))
        attn_kernel_self = ops.transpose(self.attn_kernel_self, (2, 1, 0))
        attn_kernel_neighs = ops.transpose(self.attn_kernel_neighs, (2, 1, 0))

        # Prepare message-passing
        indices = a.indices
        N = tf.shape(x, out_type=indices.dtype)[-2]
        if self.add_self_loops:
            indices = ops.add_self_loops_indices(indices, N)
        targets, sources = indices[:, 1], indices[:, 0]

        # Update node features
        x = K.dot(x, kernel)
        x = tf.reshape(x, (-1, self.attn_heads, self.channels))

        # Compute attention
        attn_for_self = tf.reduce_sum(x * attn_kernel_self, -1)
        attn_for_self = tf.gather(attn_for_self, targets)
        attn_for_neighs = tf.reduce_sum(x * attn_kernel_neighs, -1)
        attn_for_neighs = tf.gather(attn_for_neighs, sources)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)
        attn_coef = ops.unsorted_segment_softmax(attn_coef, targets, N)
        attn_coef = self.dropout(attn_coef)
        attn_coef = attn_coef[..., None]

        # Update representation
        output = attn_coef * tf.gather(x, sources)
        output = tf.math.unsorted_segment_sum(output, targets, N)

        return output, attn_coef

    def _call_dense(self, x, a):
        shape = tf.shape(a)[:-1]
        if self.add_self_loops:
            a = tf.linalg.set_diag(a, tf.ones(shape, a.dtype))
        x = tf.einsum("...NI , IHO -> ...NHO", x, self.kernel)
        attn_for_self = tf.einsum("...NHI , IHO -> ...NHO", x, self.attn_kernel_self)
        attn_for_neighs = tf.einsum(
            "...NHI , IHO -> ...NHO", x, self.attn_kernel_neighs
        )
        attn_for_neighs = tf.einsum("...ABC -> ...CBA", attn_for_neighs)

        attn_coef = attn_for_self + attn_for_neighs
        attn_coef = tf.nn.leaky_relu(attn_coef, alpha=0.2)

        mask = tf.where(a == 0.0, -10e9, 0.0)
        mask = tf.cast(mask, dtype=attn_coef.dtype)
        attn_coef += mask[..., None, :]
        attn_coef = tf.nn.softmax(attn_coef, axis=-1)
        attn_coef_drop = self.dropout(attn_coef)

        output = tf.einsum("...NHM , ...MHI -> ...NHI", attn_coef_drop, x)

        return output, attn_coef

    @property
    def config(self):
        return {
            "channels": self.channels,
            "attn_heads": self.attn_heads,
            "concat_heads": self.concat_heads,
            "dropout_rate": self.dropout_rate,
            "return_attn_coef": self.return_attn_coef,
            "attn_kernel_initializer": initializers.serialize(
                self.attn_kernel_initializer
            ),
            "attn_kernel_regularizer": regularizers.serialize(
                self.attn_kernel_regularizer
            ),
            "attn_kernel_constraint": constraints.serialize(
                self.attn_kernel_constraint
            ),
        }

