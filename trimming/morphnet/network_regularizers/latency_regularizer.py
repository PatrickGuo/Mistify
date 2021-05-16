"""A NetworkRegularizer that targets inference latency."""

from typing import Type, List

from trimming.morphnet.framework import batch_norm_source_op_handler
from trimming.morphnet.framework import conv2d_transpose_source_op_handler as conv2d_transpose_handler
from trimming.morphnet.framework import conv_source_op_handler as conv_handler
from trimming.morphnet.framework import generic_regularizers
from trimming.morphnet.framework import matmul_source_op_handler as matmul_handler

from trimming.morphnet.framework import op_handler_decorator
from trimming.morphnet.framework import op_handlers
from trimming.morphnet.framework import op_regularizer_manager as orm
from trimming.morphnet.network_regularizers import cost_calculator
from trimming.morphnet.network_regularizers import logistic_sigmoid_regularizer
from trimming.morphnet.network_regularizers import resource_function
import tensorflow.compat.v1 as tf


class LogisticSigmoidLatencyRegularizer(
    logistic_sigmoid_regularizer.LogisticSigmoidRegularizer):
  """A LogisticSigmoidRegularizer that targets Latency.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      batch_size: Integer batch size to calculate cost/loss for.
      regularize_on_mask: Bool. If True uses the binary mask as the
        regularization vector. Else uses the probability vector.
      alive_threshold: Float. Threshold below which values are considered dead.
        This can be used both when mask_as_alive_vector is True and then the
        threshold is used to binarize the sampled values and
        when mask_as_alive_vector is False, and then the threshold is on the
        channel probability.
      mask_as_alive_vector: Bool. If True use the thresholded sampled mask
        as the alive vector. Else, use thresholded probabilities from the
        logits.
      regularizer_decorator: A string, the name of the regularizer decorators to
        use. Supported decorators are listed in
        op_regularizer_decorator.SUPPORTED_DECORATORS.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for more
        detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
  """

  def __init__(
      self,
      output_boundary: List[tf.Operation],
      hardware,
      batch_size=1,
      regularize_on_mask=True,
      alive_threshold=0.1,
      mask_as_alive_vector=True,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      input_boundary: List[tf.Operation] = None,
      force_group=None,
      regularizer_blacklist=None):

    self._hardware = hardware
    self._batch_size = batch_size

    super().__init__(
        output_boundary=output_boundary,
        regularize_on_mask=regularize_on_mask,
        alive_threshold=alive_threshold,
        mask_as_alive_vector=mask_as_alive_vector,
        regularizer_decorator=regularizer_decorator,
        decorator_parameters=decorator_parameters,
        input_boundary=input_boundary,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)

  def get_calculator(self):
    return cost_calculator.CostCalculator(
        self._manager, resource_function.latency_function_factory(
            self._hardware, self._batch_size))

  @property
  def name(self):
    return 'LogisticSigmoidLatency'

  @property
  def cost_name(self):
    return self._hardware + ' Latency'


class GammaLatencyRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets latency using Gamma L1."""

  def __init__(
      self,
      output_boundary: List[tf.Operation],
      gamma_threshold,
      hardware,
      batch_size=1,
      regularizer_decorator: Type[generic_regularizers.OpRegularizer] = None,
      decorator_parameters=None,
      input_boundary: List[tf.Operation] = None,
      force_group=None,
      regularizer_blacklist=None) -> None:
    """Creates a GammaLatencyRegularizer object.

    Latency cost and regularization loss is calculated for a specified hardware
    platform.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      gamma_threshold: A float scalar, will be used as a 'gamma_threshold' for
        all instances GammaL1Regularizer created by this class.
      hardware: String name of hardware platform to target.  Must be a key from
        resource_function.PEAK_COMPUTE.
      batch_size: Integer batch size to calculate cost/loss for.
      regularizer_decorator: A string, the name of the regularizer decorators
        to use. Supported decorators are listed in
        op_regularizer_decorator.SUPPORTED_DECORATORS.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for
        more detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
    """
    source_op_handler = batch_norm_source_op_handler.BatchNormSourceOpHandler(
        gamma_threshold)
    if regularizer_decorator:
      source_op_handler = op_handler_decorator.OpHandlerDecorator(
          source_op_handler, regularizer_decorator,
          decorator_parameters)
    op_handler_dict = op_handlers.get_gamma_op_handler_dict()
    op_handler_dict.update({
        'FusedBatchNorm': source_op_handler,
        'FusedBatchNormV2': source_op_handler,
        'FusedBatchNormV3': source_op_handler,
    })

    self._manager = orm.OpRegularizerManager(
        output_boundary, op_handler_dict, input_boundary=input_boundary,
        force_group=force_group, regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager,
        resource_function.latency_function_factory(hardware, batch_size))
    self._hardware = hardware

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'Latency'

  @property
  def cost_name(self):
    return self._hardware + ' Latency'


class GroupLassoLatencyRegularizer(generic_regularizers.NetworkRegularizer):
  """A NetworkRegularizer that targets Latency using L1 group lasso."""

  def __init__(self,
               output_boundary,
               threshold,
               hardware,
               batch_size=1,
               l1_fraction=0,
               regularizer_decorator=None,
               decorator_parameters=None,
               input_boundary=None,
               force_group=None,
               regularizer_blacklist=None):
    """Creates a GroupLassoFlopsRegularizer object.

    Args:
      output_boundary: An OpRegularizer will be created for all these
        operations, and recursively for all ops they depend on via data
        dependency that does not involve ops from input_boundary.
      threshold: A float scalar, will be used as a 'threshold' for all
        regularizer instances created by this class.
      hardware: String name of hardware platform to target. Must be a key from
        resource_function.PEAK_COMPUTE.
      batch_size: Integer batch size to calculate cost/loss for.
      l1_fraction: Relative weight of L1 in L1 + L2 regularization.
      regularizer_decorator: A class of OpRegularizer decorator to use.
      decorator_parameters: A dictionary of parameters to pass to the decorator
        factory. To be used only with decorators that requires parameters,
        otherwise use None.
      input_boundary: A list of ops that represent the input boundary of the
        subgraph being regularized (input boundary is not regularized).
      force_group: List of regex for ops that should be force-grouped.  Each
        regex corresponds to a separate group.  Use '|' operator to specify
        multiple patterns in a single regex. See op_regularizer_manager for more
        detail.
      regularizer_blacklist: List of regex for ops that should not be
        regularized. See op_regularizer_manager for more detail.
    """
    custom_handlers = {
        'Conv2D':
            conv_handler.ConvSourceOpHandler(threshold, l1_fraction),
        'Conv3D':
            conv_handler.ConvSourceOpHandler(threshold, l1_fraction),
        'Conv2DBackpropInput':
            conv2d_transpose_handler.Conv2DTransposeSourceOpHandler(
                threshold, l1_fraction),
        'MatMul':
            matmul_handler.MatMulSourceOpHandler(threshold, l1_fraction)
    }
    if regularizer_decorator:
      for key in custom_handlers:
        custom_handlers[key] = op_handler_decorator.OpHandlerDecorator(
            custom_handlers[key], regularizer_decorator, decorator_parameters)

    op_handler_dict = op_handlers.get_group_lasso_op_handler_dict()
    op_handler_dict.update(custom_handlers)

    self._manager = orm.OpRegularizerManager(
        output_boundary,
        op_handler_dict,
        input_boundary=input_boundary,
        force_group=force_group,
        regularizer_blacklist=regularizer_blacklist)
    self._calculator = cost_calculator.CostCalculator(
        self._manager,
        resource_function.latency_function_factory(hardware, batch_size))
    self._hardware = hardware

  def get_regularization_term(self, ops=None):
    return self._calculator.get_regularization_term(ops)

  def get_cost(self, ops=None):
    return self._calculator.get_cost(ops)

  @property
  def op_regularizer_manager(self):
    return self._manager

  @property
  def name(self):
    return 'Latency'

  @property
  def cost_name(self):
    return self._hardware + ' Latency'
