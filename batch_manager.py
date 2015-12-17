import random

import numpy as np

class BatchManager (object):
	THRESHOLD = 0.01

	def __init__ (self, inputs, rewards):
		self.inputs = inputs
		self.rewards = rewards
		
	def create_logits (self, r_a_, r_b_):
		r_a = [0 for a in range (len (r_a_))]
		r_b = [0 for a in range (len (r_a_))]

		for i in range (len (r_a_)):
			if r_a_ [i] == 0:
				r_a [i] = 0.01
			else:
				r_a [i] = r_a_ [i]

			if r_b_ [i] == 0:
				r_b [i] = 0.01
			else:
				r_b [i] = r_b_ [i]

		r_a = np.array (r_a)
		r_b = np.array (r_b)

		dif = np.sum (np.square (np.log (r_a / r_b)))
		return np.array ([1.0 * (dif < BatchManager.THRESHOLD), 1.0 * (dif >= BatchManager.THRESHOLD)])

	def get_batch (self, batch_size):
		in_a = []
		in_b = []
		logits = []

		ids_a = random.sample (xrange (len (self.inputs)), batch_size)
		ids_b = random.sample (xrange (len (self.inputs)), batch_size)

		for i in range (len (ids_a)):
			in_a += [self.inputs [ids_a [i]]]
			in_b += [self.inputs [ids_b [i]]]
			logits += [self.create_logits (self.rewards [ids_a [i]], self.rewards [ids_b [i]])]

		return np.vstack (in_a), np.vstack (in_b), np.vstack (logits)

