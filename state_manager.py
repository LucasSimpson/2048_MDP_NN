import numpy as np

# defines a hashable board state. States are immutable
class State (object):
	def __init__ (self, state):
		self.state = tuple (state)

	def get_state (self):
		return np.array (self.state)

	def __hash__ (self):
		return hash (self.state)

	def __eq__ (self, other):
		if len (self.state) != len (other.state):
			return False

		for i in range (len (self.state)):
			if self.state [i] != other.state [i]:
				return False

		return True

	def __str__ (self):
		return '<State %s>' % str (self.state)

# defines rewards for each possible action (up, down, left, right)
class RewardInfo (object):
	# some static values for stat tracking
	total_moves = 0
	informed_moves = 0
	random_moves = 0

	# moves/index conversions
	_map = ['w', 's', 'a', 'd']
	_r_map = {'w': 0, 's': 1, 'a': 2, 'd': 3}

	def __init__ (self):
		self.rewards = np.array ([-1, -1, -1, -1])
		self.freq = [0, 0, 0, 0]

	def get_rewards (self):
		return np.copy (self.rewards)

	# get weighted random decision
	def get_move (self):
		# inc total moves
		RewardInfo.total_moves += 1

		# check for non-initialized moves
		ids = []
		for i in range (len (self.rewards)):
			if self.rewards [i] == -1:
				ids += [i]

		# if any are -1, pick 1 randomly
		if len (ids):
			RewardInfo.random_moves += 1
			return self._map [np.random.choice (ids, size=1)]

		# pick a move based on good reward for that move is
		RewardInfo.informed_moves += 1

		# sum reward for PMF creation
		sum_ = 1.0 * np.sum (self.rewards)

		# if sum is 0, just do equal distribution
		if sum_ == 0:
			return self._map [np.random.choice ([0, 1, 2, 3], size=1)]

		# use PMF for choice selection
		probs = self.rewards / sum_
		return self._map [np.random.choice ([0, 1, 2, 3], size=1, p=probs)]

	# update reward
	def update (self, move, gain):
		# get id of move
		id_ = self._r_map [move]

		# track how many times this move has been met
		self.freq [id_] += 1

		# update gain with continuous average
		f = self.freq [id_]
		self.rewards [id_] = 1.0 * ((f - 1) * self.rewards [id_] + gain) / f

	# boolean return of if self has sufficient data for use
	def is_complete (self):
		# if any rewards are not initialized (-1) return False
		for r in self.rewards:
			if r == -1:
				return False
		return True

	def __str__ (self):
		return '<RewardInfo %s>' % (self.rewards)


# handles storage/retrieval of states in master dictionairy
class StateManager (object):
	# static values for stats tracking
	dif_states = 0

	def __init__ (self):
		self.states = {}

	# lookup reward info. if not found, initialize it as default 
	def lookup (self, state):
		# lookup state
		r = self.states.get (state, None)

		# if not present, make new reward info, enter it into dict
		if not r:
			StateManager.dif_states += 1
			r = RewardInfo ()
			self.states [state] = r

		# return reward info 
		return r

	# returns all (states,reward info) pairs that have no -1 values
	def get_complete_pairs (self):
		states_ = []
		rewards = []
		for s in self.states:
			r = self.lookup (s)
			if r.is_complete ():
				states_ += [s]
				rewards += [r]
		return (states_, rewards)

	# returns all (state, reward info) pairs that have no -1 values prepped for NN processing
	def get_complete_pairs_prepped (self):
		states_ = []
		rewards = []
		for s in self.states:
			r = self.lookup (s)
			if r.is_complete ():
				states_ += [s.get_state ()]
				rewards += [r.get_rewards ()]
		return (np.vstack (states_), np.vstack (rewards))