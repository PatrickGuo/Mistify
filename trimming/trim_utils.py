import os

import tensorflow as tf
from tensorflow.python.keras.models import Model
import tensorflow.python.keras as keras


def modify_layer(model, layer_names, insert_layer_factory, width_params, insert_layer_name=None, position='replace'):
    """
    Modify (an) existing layer(s) of a model. Do not call too many times, which might cause memory leak. (Replaced
    obsolete layers have a risk of not garbage collected immediately)
    :param model: the original model
    :param layer_names: names of the layer to be selected for replacement
    :param insert_layer_factory: function to construct new layers
    :param width_params: the arguments to the insert_layer_factory(...)
    :param insert_layer_name: name of the new layer, set to None if replacing more than 1 layer
    :param position: replace or after
    :return: A brand-new model with layers replaced.
    """
    # Auxiliary dictionary to describe the network graph
    network_dict = {'input_layers_of': {}, 'new_output_tensor_of': {}}

    # Set the input layers of each layer
    # Note: Suppose Ia, Ib are two inputs, passed to a shared layer L, and output Oa, Ob respectively,
    # E.g. for out1 = layer([in1, in2, in3]), a node X is created as X -> {in1_src_layer:inbound_node[i1]:output[j1], in2's layer..., in3's layer...}
    # In summary, LAYER instance represents OP-LOGIC, NODE instance represents OP-TENSOR-CONNECTIVITY.

    for layer in model.layers:
        for node in layer.outbound_nodes:
            layer_name = node.outbound_layer.name
            if layer_name not in network_dict['input_layers_of']:
                network_dict['input_layers_of'].update(
                        {layer_name: [layer.name]})
            else:
                network_dict['input_layers_of'][layer_name].append(layer.name)

    # Set the output tensor of the input layer
    # Following the idea of: tensor <-> edge; op <-> node
    network_dict['new_output_tensor_of'].update(
            {model.layers[0].name: model.input})

    # Iterate over all layers after the input
    for layer in model.layers[1:]:

        # Determine input tensors: first find predecessor ops, then fetch output tensors of each precedessor op.
        # (A) -> Ta -> (B). We first find all {A}s, and then find all input tensor {T}s based on {A}s.
        layer_input = [network_dict['new_output_tensor_of'][layer_aux]
                       for layer_aux in network_dict['input_layers_of'][layer.name]]
        if len(layer_input) == 1:
            layer_input = layer_input[0]

        # Modify layer if name matches the given list of names
        if layer.name in layer_names:
            if position == 'replace':
                x = layer_input
            elif position == 'after':
                x = layer(layer_input)
            else:
                raise ValueError('position must be: after or replace')

            new_layer = insert_layer_factory(layer, width_params)

            if insert_layer_name:
                new_layer.name = insert_layer_name
            else:
                new_layer.name = '{}_{}_{}'.format(new_layer.name, position, layer.name)

            x = new_layer(x)
            layer_names.remove(layer.name)
        else:
            x = layer(layer_input)

        # Set new output tensor (the original one, or the one of the inserted
        # layer). We do not place the new_layer's name here because of the
        # forward ref index does not include this new layer op node.
        network_dict['new_output_tensor_of'].update({layer.name: x})

    return Model(inputs=model.inputs, outputs=x)


def layer_factory(orig_layer: keras.layers.Layer, trim_width):
    """
    The weights are not copied.
    -- How are weights stored and represented in TF?
    Ans: each weight exists as a TF variable. The actual value of the variables can be fetched/stored by calling corresponding
    operators from a default session.
    -- How to get and assign weights?
    Ans: l.get_weights() & l.set_weights(). By doing l.weight, we get the tf variables
    corresponding to these weights, and then get/set_weights() use backend to fetch or assign actual values from/to these variable tensors.
    -- How are the weight variable created for each layer?
    Ans: Each Layer class has a build() function, which will create weights tf variables specifically. build() func is
    called by Layer.__call__(...) function via _might_build(), an internal wrapper of build() func.
    -- How are weight variables associated to the tf graph?
    Ans: In build() func, it calls self.bias = self.add_weight(...). add_weight() func will create the actual TF var for
    for this weight, and then calls backend.track_variable(variable) to let the TF graph that current layer belongs to
    keep track of this weight.

    The connections are also not copied. Should modify or extract info at keras.models.Network part.
    :param orig_layer:
    :param trim_width:
    :return:
    """
    conf = orig_layer.get_config()
    if isinstance(orig_layer, keras.layers.convolutional.Conv):
        conf['filters'] = trim_width[orig_layer.name]
    elif isinstance(orig_layer, keras.layers.Dense):
        conf['units'] = trim_width[orig_layer.name]
    return type(orig_layer).from_config(conf)


def do_trim(inputs, logits, labels, configs,
            pattern='flops', hw='K80', reg_strength=None, max_steps=1000000, trim_dir='trim_ckpt'):
    """
    Do the actual trimming. Keypoints: 1) parsing configs and make it some order, 2) stop and checkpoint if satisfied.
    :param inputs:
    :param logits:
    :param labels:
    :param configs: dict[String, (CPU, MEM, LATENCY)], organized from loose to tight
    :param pattern:
    :param hw:
    :param reg_strength:
    :param max_steps:
    :param trim_dir: 
    :return:
    """
    from trimming.morphnet.network_regularizers import flop_regularizer, latency_regularizer, model_size_regularizer
    from trimming.morphnet.tools import structure_exporter

    # inputs, labels = preprocessor()
    # logits = build_model(inputs, labels, ...)

    idx = 0
    if pattern == 'flops':
        network_regularizer = flop_regularizer.GammaFlopsRegularizer(
            output_boundary=[logits.op],
            input_boundary=[inputs.op, labels.op],
            gamma_threshold=1e-3
        )
        idx = 0
    elif pattern == 'latency':
        network_regularizer = latency_regularizer.GammaLatencyRegularizer(
            output_boundary=[logits.op],
            input_boundary=[inputs.op, labels.op],
            gamma_threshold=1e-3,
            hardware=hw
        )
        idx = 2
    else:
        network_regularizer = model_size_regularizer.GammaModelSizeRegularizer(
            output_boundary=[logits.op],
            input_boundary=[inputs.op, labels.op],
            gamma_threshold=1e-3
        )
        idx = 1
    regularization_strength = reg_strength if reg_strength else 1 / network_regularizer.get_cost().eval()
    regularizer_loss = (network_regularizer.get_regularization_term() * regularization_strength)

    model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

    train_op = optimizer.minimize(model_loss + regularizer_loss)

    tf.summary.scalar(network_regularizer.cost_name, network_regularizer.get_cost())

    exporter = structure_exporter.StructureExporter(
        network_regularizer.op_regularizer_manager)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for step in range(max_steps):
            _, cost, structure_exporter_tensors = sess.run([train_op, network_regularizer.get_cost(), exporter.tensors])
            # structure_exporter_tensors is a tensor with size=units/filters, each element is 1/0
            # This can be used for loading pre-trained filters.
            for name in configs.keys():  # only write for the appropriate one
                if cost < 1.2 * configs[name][idx]:
                    exporter.populate_tensor_values(structure_exporter_tensors)
                    with tf.gfile.Open(os.path.join(trim_dir, name), 'w') as f:
                        exporter.save_alive_counts(f)
                configs.pop(name)
                break
            if step % 1000 == 0:
                exporter.populate_tensor_values(structure_exporter_tensors)
                exporter.create_file_and_save_alive_counts(trim_dir, step)
