# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain Adaptation Loss Functions.

The following domain adaptation loss functions are defined:

- Maximum Mean Discrepancy (MMD).
  Relevant paper:
    Gretton, Arthur, et al.,
    "A kernel two-sample test."
    The Journal of Machine Learning Research, 2012

- Correlation Loss on a batch.
"""
from functools import partial
import tensorflow as tf


import utils
slim = tf.contrib.slim


################################################################################
# SIMILARITY LOSS
################################################################################
def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, loss_value)


  return loss_value


def correlation_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the correlation between two representations.

  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight

  assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
  with tf.control_dependencies([assert_op]):
    tag = 'Correlation Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, corr_loss)
    tf.losses.add_loss(corr_loss)

  return corr_loss


def Deep_CORAL_loss(source_samples, target_samples, weight, scope=None):

  with tf.name_scope('Deep_CORAL_loss'):
    d=source_samples.get_shape().as_list()[1]
    
    source_samples = tf.reduce_mean(source_samples, 0)-source_samples
    source_s=tf.transpose(source_samples) @ source_samples
    
    target_samples = tf.reduce_mean(target_samples, 0)-target_samples
    target_s=tf.transpose(target_samples) @ target_samples


    corr_loss = tf.reduce_mean(tf.multiply((source_s-target_s), (source_s-target_s)))* weight
    corr_loss = corr_loss/(4*d*d)
    tf.losses.add_loss(corr_loss)

  return corr_loss

def dann_loss(source_samples, target_samples, weight, scope=None):
  """Adds the domain adversarial (DANN) loss.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.variable_scope('dann'):
    batch_size = tf.shape(source_samples)[0]
    samples = tf.concat(axis=0, values=[source_samples, target_samples])
    samples = slim.flatten(samples)

    domain_selection_mask = tf.concat(
        axis=0, values=[tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))])

    # Perform the gradient reversal and be careful with the shape.
    grl = grl_ops.gradient_reversal(samples)
    grl = tf.reshape(grl, (-1, samples.get_shape().as_list()[1]))

    grl = slim.fully_connected(grl, 100, scope='fc1')
    logits = slim.fully_connected(grl, 1, activation_fn=None, scope='fc2')

    domain_predictions = tf.sigmoid(logits)

  domain_loss = tf.losses.log_loss(
      domain_selection_mask, domain_predictions, weights=weight)

  domain_accuracy = utils.accuracy(
      tf.round(domain_predictions), domain_selection_mask)

  assert_op = tf.Assert(tf.is_finite(domain_loss), [domain_loss])
  with tf.control_dependencies([assert_op]):
    tag_loss = 'losses/domain_loss'
    tag_accuracy = 'losses/domain_accuracy'
    if scope:
      tag_loss = scope + tag_loss
      tag_accuracy = scope + tag_accuracy

    tf.summary.scalar(tag_loss, domain_loss)
    tf.summary.scalar(tag_accuracy, domain_accuracy)

  return domain_loss


