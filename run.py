import random
import numpy as np

import matplotlib.pyplot as plt

from Game2048 import Game


# some global stats to track
dif_states = 0
moves_total = 0
random_moves = 0
informed_moves = 0

# defines a hashable board state. States are immutable
class State (object):
	def __init__ (self, state):
		self.state = tuple (state)

	def __hash__ (self):
		return hash (self.state)

	def __eq__ (self, other):
		if len (self.state) != len (other.state):
			return False

		for i in range (len (self.state)):
			if self.state [i] != other.state [i]:
				return False

		return True

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

# transorm board into state, grouping similar board into same state
# this is dope, works good for initial turns but sucks after
def convolute (board):
	state = []
	for a in range (0, 3):
		for b in range (0, 3):
			high = 0
			for c in range (0, 2):
				for d in range (0, 2):
					e = board [a * 4 + b + c * 4 + d]
					if e > high:
						high = e
			state += [high]
	return state

g = Game ()
sm = StateManager ()

scores = []
running_length = 100
averages = []
num_games = int (1 * 1e3)

print 'Playing %s games.' % num_games



while (num_games >= 0):
	while (not g.is_stale ()):
		# get current states reward info
		s = State (g.get_state ())

		# save current score
		current_score = g.get_score ()

		# get reward info for current state
		r = sm.lookup (s)

		# get move decision
		move = r.get_move ()

		# make move
		g.process_move (move)

		# get new score
		new_score = g.get_score ()

		# calc score gain
		gain = new_score - current_score

		# update reward info
		#r.update (move, gain)

	
	scores += [g.get_score ()]
	if len (scores) % running_length  == 0:
		print 'Running average: ', sum (scores [-running_length:]) / running_length
		print 'Dif states:    ', StateManager.dif_states
		print 'Total moves:   ', RewardInfo.total_moves
		print 'random moves:   %s (%s)' % (RewardInfo.random_moves, 1.0 * RewardInfo.random_moves / RewardInfo.total_moves)
		print 'informed moves: %s (%s)' % (RewardInfo.informed_moves, 1.0 * RewardInfo.informed_moves / RewardInfo.total_moves)
		print ''

	if len (scores) > 100:
		averages += [sum (scores [-100:]) / 100]
	else:
		averages += [sum (scores) / len (scores)]

	g = Game ()
	num_games -= 1


plt.plot (scores)
plt.ylabel ('Game scores')
plt.xlabel ('Trial #')
plt.show ()
