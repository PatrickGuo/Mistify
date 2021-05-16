"""
Implementation of AdaptExec class.
-- wrapping around MorphNet as an example.
"""

import os
import tensorflow as tf
from datetime import datetime
from trimming.morphnet.network_regularizers import flop_regularizer, latency_regularizer, model_size_regularizer
from trimming.morphnet.tools import structure_exporter
from adapt_utils import select_keras_base_model


class AdaptExec(object):

    def __init__(self,
                 job_id,
                 conf_id,
                 base_model_name="ResNet50",
                 base_model_path='',
                 num_classes=10,
                 learning_rate=1e-3,
                 batch_size=256,
                 num_gpus=4,
                 main_train_device="/cpu:0",
                 main_eval_device="/gpu:0",
                 morphnet_regularizer_algorithm="GroupLasso",
                 morphnet_target_costs=["FLOPs"],
                 morphnet_target_cost_thresholds=[1.0],
                 morphnet_hardware="V100",
                 morphnet_regularizer_threshold=1e-2,
                 morphnet_regularization_strength=1e-9,
                 log_dir="./morphnet_log"):
        """
        Initialize AdaptExec instance.
        Args:
            base_model_name: Keras model class.
            num_classes: Number of classes for classification. Integer.
            learning_rate: Learning rate. Float.
            batch_size: Batch size for multi-GPU training. The batch size would be further divided across multiple GPUs. Integer.
            num_gpus: Number of GPU devices used for model training. Integer.
            main_train_device: The GPU device used for computing the average of gradients collected from all the GPU devices. For multi-GPU training, on my DGX-station (E5-2698, 4x V100), it seems setting it to "/cpu:0" is the fastest. String.
            main_eval_device: The GPU device used for evaluation and inference. String.
            morphnet_regularizer_algorithm: MorphNet regularization algorithm name. Currently we support "GroupLasso" and "Gamma". String.
            morphnet_target_costs: The optimization target cost for MorphNet regularization algorithms. Currently we support "FLOPs", "Latency", and "ModelSize". String.
            morphnet_target_cost_thresholds: The weight of the optimization targets
            morphnet_hardware: The hardware for using "Latency" as the optimization target cost. Currently we support "V100" and "P100". String.
            morphnet_regularizer_threshold: The threshold determines which output channels can be eliminated. Float.
            morphnet_regularization_strength: The regularization strength for MorphNet as MorphNet is a regularization technique. Float.
            log_dir: The log directory for TensorBoard, intermediate files, and model architecture files.
        """
        self.job_id = job_id
        self.conf_id = conf_id
        self.base_model = select_keras_base_model(base_model_name=base_model_name)
        self.base_model_path = base_model_path
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.main_train_device = main_train_device
        self.main_eval_device = main_eval_device
        self.morphnet_regularizer_algorithm = morphnet_regularizer_algorithm
        self.morphnet_target_costs = morphnet_target_costs
        self.morphnet_target_cost_thresholds = morphnet_target_cost_thresholds
        self.morphnet_hardware = morphnet_hardware

        self.morphnet_regularizer_threshold = morphnet_regularizer_threshold
        # Setting regularization strength to zero removes MorphNet architecture search.
        self.morphnet_regularization_strength = morphnet_regularization_strength

        self.log_dir = log_dir
        self.global_step = 0

        self.initialize()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.log_dir = os.path.join(self.log_dir,
                                    datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.writer = tf.summary.FileWriter(self.log_dir,
                                            tf.get_default_graph())
        self.morphnet_summary = self.summary()

    def config_model(self, input_tensor=None):
        """
        Initialize the model.
        """
        with tf.device(self.main_train_device):

            base_model = self.base_model(weights=None,
                                         include_top=False,
                                         input_tensor=input_tensor)
            x = base_model.output
            # Add a global spatial average pooling layer since MorphNet does not support Flatten/Reshape OPs.
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(1024, activation="relu")(x)
            logits = tf.keras.layers.Dense(self.num_classes)(x)

            self.model = tf.keras.Model(inputs=base_model.input, outputs=logits)
            # When we have an existing snapshot to start
            if self.base_model_path is not "":
                self.model = tf.keras.models.load_model(self.base_model_path)

            self.inputs = self.model.input
            self.labels = tf.placeholder(tf.float32, [None, self.num_classes])

            self.morphnet_regularization_strength_placeholder = tf.placeholder(
                tf.float32, shape=[])

    def embed_morphnet(self, input_boundary, output_boundary,
                       morphnet_regularization_strength,
                       morphnet_cost_thresholds):
        """
        Initialize MorphNet components.
        """
        # all the
        network_regularizers = []
        if self.morphnet_regularizer_algorithm == "GroupLasso":
            if "FLOPs" in self.morphnet_target_costs:
                regularizer_fn = flop_regularizer.GroupLassoFlopsRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    threshold=self.morphnet_regularizer_threshold))
            if "Latency" in self.morphnet_target_costs:
                if self.morphnet_hardware not in ("V100", "P100"):
                    raise Exception(
                        "Unsupported MorphNet Hardware For Latency Regularizer!"
                    )
                regularizer_fn = latency_regularizer.GroupLassoLatencyRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    threshold=self.morphnet_regularizer_threshold,
                    hardware=self.morphnet_hardware))
            if "ModelSize" in self.morphnet_target_costs:
                regularizer_fn = model_size_regularizer.GroupLassoModelSizeRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    threshold=self.morphnet_regularizer_threshold))
        elif self.morphnet_regularizer_algorithm == "Gamma":
            if "FLOPs" in self.morphnet_target_costs:
                regularizer_fn = flop_regularizer.GammaFlopsRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    gamma_threshold=self.morphnet_regularizer_threshold))
            elif "Latency" in self.morphnet_target_costs:
                if self.morphnet_hardware not in ("V100", "P100"):
                    raise Exception(
                        "Unsupported MorphNet Hardware For Latency Regularizer!"
                    )
                regularizer_fn = latency_regularizer.GammaLatencyRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    gamma_threshold=self.morphnet_regularizer_threshold,
                    hardware=self.morphnet_hardware))
            elif "ModelSize" in self.morphnet_target_costs:
                regularizer_fn = model_size_regularizer.GammaModelSizeRegularizer
                network_regularizers.append(regularizer_fn(
                    output_boundary=output_boundary,
                    input_boundary=input_boundary,
                    gamma_threshold=self.morphnet_regularizer_threshold))
        else:
            raise Exception("Unsupported MorphNet Regularizer Algorithm!")

        # overall total structure loss tensor
        regularizer_loss = sum([netreg.get_regularization_term() * morphnet_regularization_strength
                                * morphnet_cost_thresholds[index]
                                for index, netreg in enumerate(network_regularizers)])
        exporter = structure_exporter.StructureExporter(network_regularizers[0].op_regularizer_manager)
        costs = [r.get_cost() for r in network_regularizers]

        return network_regularizers, regularizer_loss, exporter, costs

    def construct_loss(self, logits, labels):
        """
        Initialize the loss of the original model. The MorphNet regularization loss is not included.
        Args:
            logits: Logits output tensor from the model. Tensor.
            labels: One-hot encoded ground-truth label tensor. Tensor.
        Returns:
            model_loss: The loss of original model. Tensor.
        """
        model_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels,
                                                       logits=logits))
        return model_loss

    def average_gradients(self, tower_grads):
        """
        Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            # ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)

        return average_grads

    def config_training_instance(self):
        """
        Initialize the single-node multi-GPU training instance.
        """
        # Compute the average of the gradients main_train_device
        tower_grads = []

        # Distribute the model onto available GPUs
        for i in range(self.num_gpus):
            with tf.device("/gpu:{}".format(i)):

                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate)
                batch_size_instance = self.batch_size // self.num_gpus

                # Split data between GPUs
                inputs_instance = self.inputs[i * batch_size_instance:(i + 1) *
                                              batch_size_instance]
                labels_instance = self.labels[i * batch_size_instance:(i + 1) *
                                              batch_size_instance]

                logits = self.model(inputs_instance)
                trainable_variables = self.model.trainable_variables
                model_loss = self.construct_loss(logits=logits,
                                                 labels=labels_instance)
                network_regularizers, regularizer_loss, exporter, costs = self.embed_morphnet(
                    input_boundary=[inputs_instance.op],
                    output_boundary=[logits.op],
                    morphnet_regularization_strength=self.
                    morphnet_regularization_strength_placeholder,
                    morphnet_cost_thresholds=self.morphnet_target_cost_thresholds)
                total_loss = model_loss + regularizer_loss

                grads = optimizer.compute_gradients(
                    total_loss, var_list=trainable_variables)
                tower_grads.append(grads)

                # Usually we would use the first GPU
                if i == 0:
                    # Evaluate model (with test logits, for dropout to be disabled)
                    self.logits_train_instance = logits
                    self.model_loss_train_instance = model_loss
                    self.probs_train_instance = tf.nn.softmax(logits)
                    self.correct_pred_train_instance = tf.equal(
                        tf.argmax(logits, 1), tf.argmax(labels_instance, 1))
                    self.accuracy_train_instance = tf.reduce_mean(
                        tf.cast(self.correct_pred_train_instance, tf.float32))

                    self.network_regularizer_train_instance = network_regularizers
                    self.regularizer_loss_train_instance = regularizer_loss
                    self.total_loss_train_instance = total_loss
                    self.exporter_train_instance = exporter
                    self.cost_train_instance = costs

        # Compute the average of the gradients main_train_device
        with tf.device(self.main_train_device):
            grads = self.average_gradients(tower_grads)
            self.train_op = optimizer.apply_gradients(grads, global_step=None)

    def config_evaluation_instance(self):
        """
        Initialize the single-GPU evaluation instance.
        """
        with tf.device(self.main_eval_device):
            self.inputs_eval = self.model.input
            self.labels_eval = tf.placeholder(tf.float32,
                                              [None, self.num_classes])
            self.logits_eval = self.model(self.inputs_eval)
            self.probs_eval = tf.nn.softmax(self.logits_eval)
            self.correct_pred_eval = tf.equal(tf.argmax(self.logits_eval, 1),
                                              tf.argmax(self.labels_eval, 1))
            self.accuracy_eval = tf.reduce_mean(
                tf.cast(self.correct_pred_eval, tf.float32))
            # Created for checking the MorphNet statistics only
            _, _, self.exporter_eval, self.cost_eval = self.embed_morphnet(
                input_boundary=[self.inputs_eval.op],
                output_boundary=[self.logits_eval.op],
                morphnet_regularization_strength=self.
                    morphnet_regularization_strength_placeholder,
                morphnet_cost_thresholds=self.morphnet_target_cost_thresholds)

    def validate(self, inputs, labels):
        """
        Validate one batch of the validation data using tf.Session().
        """
        # Set the phase to test.
        tf.keras.backend.set_learning_phase(0)
        accuracy = self.sess.run([self.accuracy_eval],
                                 feed_dict={
                                     self.inputs_eval: inputs,
                                     self.labels_eval: labels
                                 })
        costs = self.sess.run(self.cost_eval,
                              feed_dict={
                                   self.inputs_eval: inputs,
                                   self.labels_eval: labels
                              })
        return accuracy, costs

    def initialize(self):
        self.config_model()
        self.config_training_instance()
        self.config_evaluation_instance()

    def search_exec(self, inputs, labels):
        """
        Train one batch of the training data using tf.Session().
        """
        # Set the phase to training.
        tf.keras.backend.set_learning_phase(1)
        self.global_step += 1

        _, structure_exporter_tensors, loss, accuracy, morphnet_summary = self.sess.run(
            [
                self.train_op, self.exporter_train_instance.tensors,
                self.total_loss_train_instance, self.accuracy_train_instance,
                self.morphnet_summary
            ],
            feed_dict={
                self.inputs:
                    inputs,
                self.labels:
                    labels,
                self.morphnet_regularization_strength_placeholder:
                    self.morphnet_regularization_strength
            })
        costs = self.sess.run(
            self.cost_train_instance,
            feed_dict={
                self.inputs:
                    inputs,
                self.labels:
                    labels,
                self.morphnet_regularization_strength_placeholder:
                    self.morphnet_regularization_strength
            }
        )
        self.writer.add_summary(morphnet_summary, self.global_step)
        self.exporter_train_instance.populate_tensor_values(
            structure_exporter_tensors)

        return loss, accuracy, costs

    def export_model_config_with_inputs(self, inputs):
        """
        Export the model architecture after MorphNet optimization to JSON file.
        Require inputs to compute the model architecture.
        """
        structure_exporter_tensors = self.sess.run(
            self.exporter_eval.tensors, feed_dict={self.inputs: inputs})
        self.exporter_eval.populate_tensor_values(structure_exporter_tensors)
        path = self.exporter_eval.create_file_and_save_alive_counts(
            self.log_dir, self.global_step)
        return path

    def export_model(self):
        path = os.path.join(self.log_dir, self.job_id, self.conf_id, 'snapshot')
        self.model.save(path)
        return path

    def measure_costs(self, inputs):
        """
        Get the target cost. Require inputs to compute the target cost.
        """
        costs = self.sess.run(self.cost_eval, feed_dict={self.inputs: inputs})

        return costs

    def adjust_params(self, inputs, target_costs, original_costs):
        """
        Set the weighted MorphNet regularization strength; periodically called to calibrate.
        """
        costs = self.measure_costs(inputs)
        weights = [(target_costs[i] - costs[i]) / (target_costs[i] - original_costs[i]) for i in range(target_costs)]
        totalweights = sum(weights)
        for i in range(weights):
            self.morphnet_target_cost_thresholds[i] = weights[i] / totalweights

    def summary(self):
        """
        Create TensorFlow summaries for TensorBoard.
        """
        regularization_loss_summary = tf.summary.scalar(
            "RegularizationLoss", self.regularizer_loss_train_instance)
        summaries = [regularization_loss_summary]
        for idx, reg_instance in enumerate(self.network_regularizer_train_instance):
            summaries.append(tf.summary.scalar(
                reg_instance.cost_name,
                self.cost_train_instance[idx]))
        morphnet_summary = tf.summary.merge(summaries)
        return morphnet_summary

    def close(self):
        """
        Close the AdaptExec model instance.
        """
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
