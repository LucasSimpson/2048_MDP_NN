from colorama import *
import math, random, time

class Game (object):
	colorConstants = [Fore.BLACK, Fore.MAGENTA, Fore.RED, Fore.BLUE, Fore.CYAN, Fore.GREEN, Fore.YELLOW, Fore.WHITE]
	move_up_indices = [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
	move_left_indeces = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
	move_map = {
		'a': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
		's': [[15, 11, 7, 3], [14, 10, 6, 2], [13, 9, 5, 1], [12, 8, 4, 0]],
		'd': [[15, 14, 13, 12], [11, 10, 9, 8], [7, 6, 5, 4], [3, 2, 1, 0]],
		'w': [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]],
	}

	def __init__ (self):
		self.state = [0 for a in range (16)]
		self.score = 0
		self.stale_count = 0
		self.spawn_tile ()
		#self.spawn_tile ()

	def get_state (self):
		return self.state

	def get_score (self):
		return self.score

	def spawn_tile (self):
		validIndices = []
		for index, value in enumerate (self.state):
			if value == 0:
				validIndices += [index]

		self.state [validIndices [random.randint (0, len (validIndices) - 1)]] = 2

	def process_move (self, move, visual=False):
		hasChanged = False

		for row in self.move_map [move]:
			if self.collapse_row (row):
				hasChanged = True

		if hasChanged:
			self.stale_count = 0
			self.spawn_tile ()
		else:
			self.stale_count += 1

		if visual:
			print 'Move: ' + move
			print self
			time.sleep (0.5)

	def is_stale (self):
		return self.stale_count > 2

	def collapse_row (self, indices):
		def look_ahead (i):
			i += 1
			while (i < 4):
				if self.state [indices [i]] != 0:
					return i, self.state [indices [i]]
				i += 1
			return 0, 0

		hasChanged = False
		slide_amount = 0
		new_row = []

		for a in range (len (indices)):
			if self.state [indices [a]] != 0:
				i, val = look_ahead (a)
				if self.state [indices [a]] == val:
					new_row += [val * 2]
					self.score += val * 2
					self.state [indices [i]] = 0
				else:
					new_row += [self.state [indices [a]]]

		new_row += [0] * (4 - len (new_row))

		for a in range (len (indices)):
			if self.state [indices [a]] != new_row [a]:
				hasChanged = True
				self.state [indices [a]] = new_row [a] 

		return hasChanged

	def __str__ (self):
		r = 'Score: ' + str (self.score) + '\n'
		for index, value in enumerate (self.state):
			clr = Fore.BLACK
			if value != 0:
				k = int (math.log (value, 2)) % 8
				clr = self.colorConstants [k]

			r += Style.BRIGHT + clr + str (value) + '\t'

			if (index + 1) % 4 == 0:
				r += Style.NORMAL + Fore.WHITE + '\n'
		return r