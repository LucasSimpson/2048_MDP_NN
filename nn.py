import tensorflow as tf
import numpy as np

import random

class NeuralNetwork (object):
	SPECIES_RANGE = 0.05
	BATCH_SIZE = 300
	ACC_TARGET = 0.8

	def __init__ (self):
		self.inputs_a = tf.placeholder ("float", shape=[None, 16])
		self.inputs_b = tf.placeholder ("float", shape=[None, 16])
		self.answer_logits = tf.placeholder ("float", shape=[None, 2])

		self.readout = None
		self.train_step = None
		self.accuracy = None
		self.user_output = None

		self.create_graph ()
		self.create_training_graph ()
		self.create_accuracy_graph ()
		self.create_user_readout_graph ()

		self.sess = tf.InteractiveSession ()
		init = tf.initialize_all_variables ()
		self.sess.run (init)


	def create_graph (self):
		# shortcut to make a weight variable with truncated normal distribution
		def weight_variable (shape):
			initial = tf.truncated_normal (shape, stddev=0.1)
			return tf.Variable (initial)

		# shortcut for making bias variables with 0.1 starting constant
		def bias_variable (shape):
			initial = tf.constant (0.1, shape=shape)
			return tf.Variable (initial)

		grid_input_a = tf.reshape (self.inputs_a, [-1, 4, 4, 1])
		grid_input_b = tf.reshape (self.inputs_b, [-1, 4, 4, 1])

		filter_1 = weight_variable ([2, 2, 1, 16])
		biases_1 = bias_variable ([16])
		conv_1_a = tf.nn.conv2d (grid_input_a, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID') + biases_1
		conv_1_b = tf.nn.conv2d (grid_input_b, filter=filter_1, strides=[1, 1, 1, 1], padding='VALID') + biases_1

		relu_1_a = tf.nn.relu (conv_1_a)
		relu_1_b = tf.nn.relu (conv_1_b)

		filter_2 = weight_variable ([2, 2, 16, 32])
		biases_2 = bias_variable ([32])

		conv_2_a = tf.nn.conv2d (relu_1_a, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID') + biases_2
		conv_2_b = tf.nn.conv2d (relu_1_b, filter=filter_2, strides=[1, 1, 1, 1], padding='VALID') + biases_2

		relu_2_a = tf.nn.relu (conv_2_a)
		relu_2_b = tf.nn.relu (conv_2_b)

		side_length = 2 * 2 * 32

		lin_a = tf.reshape (relu_2_a, [-1, side_length])
		lin_b = tf.reshape (relu_2_b, [-1, side_length])

		lin_all = tf.concat (1, [lin_a, lin_b])
		lin_all_synced = tf.tuple ([lin_all]) [0]

		fc_1_w = weight_variable ([side_length * 2, 1024])
		fc_1_b = bias_variable ([1024])
		fc_1 = tf.matmul (lin_all_synced, fc_1_w) + fc_1_b

		fc_2_w = weight_variable ([1024, 2])
		fc_2_b = bias_variable ([2])
		fc_2 = tf.matmul (fc_1, fc_2_w) + fc_2_b

		self.readout = tf.nn.softmax (fc_2)

	def create_training_graph (self):
		cross_entropy = -tf.reduce_sum (self.answer_logits * tf.log (self.readout))
		self.train_step = tf.train.AdamOptimizer (1e-4).minimize (cross_entropy)

	def create_accuracy_graph (self):
		is_right = tf.equal (tf.argmax (self.readout, 1), tf.argmax (self.answer_logits, 1))
		self.accuracy = tf.reduce_mean (tf.cast (is_right, "float"))

	def create_user_readout_graph (self):
		self.user_output = tf.argmax (self.readout, 1)

	def full_eval (self, batch_manager):
		in_a, in_b, logits = batch_manager.get_batch (NeuralNetwork.BATCH_SIZE)
		feed_dict = {
			self.inputs_a: in_a,
			self.inputs_b: in_b,
			self.answer_logits: logits,
		}
		return self.sess.run ([self.accuracy], feed_dict=feed_dict) [0]


	def train (self, batch_manager):
		print 'Training:'
		acc = 0
		a = 0
		while (acc < NeuralNetwork.ACC_TARGET):
			a += 1
			in_a, in_b, logits = batch_manager.get_batch (NeuralNetwork.BATCH_SIZE)
			
			feed_dict = {
				self.inputs_a: in_a,
				self.inputs_b: in_b,
				self.answer_logits: logits,
			}

			if a % 50 == 0:
				acc = self.full_eval (batch_manager)
				print 'Iter=%s, Accuracy=%s' % (a, acc)

			self.sess.run (self.train_step, feed_dict=feed_dict)
			
		acc = self.full_eval (batch_manager)
		print 'Iter=%s, Accuracy=%s' % (a, acc)

	# Given two states, return True if both are alike
	def compute (self, in_a, in_b):

		return self.sess.run (self.user_output, feed_dict={
				self.inputs_a: np.reshape (in_a, (1, 16)),
				self.inputs_b: np.reshape (in_b, (1, 16)),
			}) [0] == 1

