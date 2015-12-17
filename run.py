import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from Game2048 import Game
from state_manager import StateManager, State, RewardInfo
from nn import NeuralNetwork
from batch_manager import BatchManager


# small wrapper around state object, representing a suite of alike states by the rep
class Species (object):
	def __init__ (self, rep):
		self.rep = rep

	# resets the rep for the species
	def set_rep (self, rep):
		self.rep = rep

	# return the representative for this species
	def get_rep (self):
		return self.rep

	# species are completely repped by self.rep, including hash and eq
	def __hash__ (self):
		return self.rep.__hash__ ()

	def __eq__ (self, other):
		return self.rep.__eq__ (other.get_rep ())

# manages job of finding species associated with various boards
class SpeciesManager (object):
	num_species = 0

	# if tf_nn not passed, then every unique board is a unique species
	def __init__ (self, tf_nn=None):
		self.species = {}
		self.tf_nn = tf_nn

	def find_species (self, state):
		# use neural network to group states
		in_ = state.get_state ()

		# check state vs every known species (TODO optimize this somehow)
		for species in self.species:
			other_ = species.get_rep ().get_state ()

			alike = self.tf_nn.compute (in_, other_)

			# if state is alike, return that species as its similar enough
			if alike:
				return species

		# return species associated with inputted state
		return s

	# given a state and reward info, inserts it into set
	def insert (self, state, r_info):
		s = self.find_species (state)

		# only create new data if species not found
		if not s:
			self.species [s] = r_info

	# given a state, output the reward info
	def lookup (self, state):
		# lookup state in species dict
		spec = self.find_species (state)

		# if not found, create a new one
		if not spec:
			spec = Species (state)
			r = RewardInfo ()
			self.species [s] = r
			return spec, r

		# return reward for inputted state
		return spec, self.species [spec]





g = Game ()
sm = StateManager ()


scores = []
random_moves = []
informed_moves = []
dif_states = []
x_data = []


running_length = 500
averages = []
num_games = int (4 * 1e3)

print 'Playing %s games.' % num_games

while (num_games >= 0):
	while (not g.is_stale ()):
		# get current states reward info
		s = State (g.get_state ())

		# save current score
		current_score = g.get_score ()

		# get reward info for current state
		r = sm.lookup (s)
		#r = spm.lookup (s)

		# get move decision
		move = r.get_move ()

		# make move
		g.process_move (move)

		# get new score
		new_score = g.get_score ()

		# calc score gain
		gain = new_score - current_score

		# update reward info
		r.update (move, gain)

	
	x_data += [len (x_data)]
	scores += [g.get_score ()]
	random_moves += [100.0 * RewardInfo.random_moves / RewardInfo.total_moves]
	informed_moves += [100.0 * RewardInfo.informed_moves / RewardInfo.total_moves]
	dif_states += [StateManager.dif_states]


	if len (scores) % running_length  == 0:
		print 'Best score thus far: ', (np.max (scores))
		print 'Running average: %s' % (sum (scores [-running_length:]) / running_length)
		print 'Num species:    %s' % (StateManager.dif_states)
		print 'Total moves:     %s' % (RewardInfo.total_moves)
		print 'random moves:    %s (%s)' % (RewardInfo.random_moves, 1.0 * RewardInfo.random_moves / RewardInfo.total_moves)
		print 'informed moves:  %s (%s)' % (RewardInfo.informed_moves, 1.0 * RewardInfo.informed_moves / RewardInfo.total_moves)
		print 'games left:      %s' % num_games
		print ''

	if len (scores) > 100:
		averages += [sum (scores [-100:]) / 100]
	else:
		averages += [sum (scores) / len (scores)]

	g = Game ()
	num_games -= 1



states, rewards = sm.get_complete_pairs_prepped ()

bm = BatchManager (states, rewards)

print ''
print 'Completed states: %s' % len (states)
print 'Percent of all states: %s' % (1.0 * len (states) / StateManager.dif_states)
print ''






# train neural network
nn = NeuralNetwork ()
spm = SpeciesManager (nn)
nn.train (bm)

states, rewards = sm.get_complete_pairs ()

print states [40]
print rewards [40]

# insert pairs
for i in range (len (states)):
	spm.insert  (states [i], rewards [i])







scores = []
random_moves = []
informed_moves = []
dif_states = []
x_data = []

running_length = 10
averages = []
num_games = int (4 * 1e3)

print 'Playing %s games.' % num_games

while (num_games >= 0):
	while (not g.is_stale ()):
		# get current states reward info
		s = State (g.get_state ())

		# save current score
		current_score = g.get_score ()

		# get reward info for current state
		s, r = spm.lookup (s)

		# get move decision
		move = r.get_move ()

		print '\n' * 10
		print 'Board:'
		print g
		print 'Species match rep:'
		g_ = Game ()
		g_.state = s.get_rep ().get_state ()
		print g_
		print 'R info: %s' % r.get_rewards ()

		print 'TF jic: %s' % nn.compute (g.get_state (), g_.get_state ())
		print 'Move: %s' % move
		print 'New Board:'

		# make move
		g.process_move (move)

		print g

		w = raw_input ('Waiting...')

		# get new score
		new_score = g.get_score ()

		# calc score gain
		gain = new_score - current_score

		# update reward info
		r.update (move, gain)

	
	x_data += [len (x_data)]
	scores += [g.get_score ()]
	random_moves += [100.0 * RewardInfo.random_moves / RewardInfo.total_moves]
	informed_moves += [100.0 * RewardInfo.informed_moves / RewardInfo.total_moves]
	dif_states += [StateManager.dif_states]


	if len (scores) % running_length  == 0:
		print 'Best score thus far: ', (np.max (scores))
		print 'Running average: %s' % (sum (scores [-running_length:]) / running_length)
		print 'Num species:    %s' % (SpeciesManager.num_species)
		print 'Total moves:     %s' % (RewardInfo.total_moves)
		print 'random moves:    %s (%s)' % (RewardInfo.random_moves, 1.0 * RewardInfo.random_moves / RewardInfo.total_moves)
		print 'informed moves:  %s (%s)' % (RewardInfo.informed_moves, 1.0 * RewardInfo.informed_moves / RewardInfo.total_moves)
		print 'games left:      %s' % num_games
		print ''

	if len (scores) > 100:
		averages += [sum (scores [-100:]) / 100]
	else:
		averages += [sum (scores) / len (scores)]

	g = Game ()
	num_games -= 1