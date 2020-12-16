#!/usr/bin/env python
import argparse
import lz4.frame
import numpy as np
import scipy.spatial
import sys

class RecursionLimitManager:
	def __init__(self, lim):
		self.lim = lim
	def __enter__(self):
		self.oldlim = sys.getrecursionlimit()
		sys.setrecursionlimit(self.lim)
	def __exit__(self, type, value, traceback):
		sys.setrecursionlimit(self.oldlim)

def recursionlimit(lim):
	return RecursionLimitManager(lim)

def project_words(words, good, evil, lawful, chaotic):
	neutral_morality = (good + evil) / 2
	neutral_order = (lawful + chaotic) / 2
	basis = np.identity(words.shape[0])
	basis[:,0:1] = good - neutral_morality
	basis[:,1:2] = lawful - neutral_morality
	basis[:,2:3] = neutral_order - neutral_morality

	projected = np.linalg.inv(basis)[0:2,:] @ (words - neutral_morality)
	return projected

def find_most(base, words):
	mag_base = np.linalg.norm(base)
	mag_words = np.linalg.norm(words, axis=1)
	proj = words * base
	sum = proj.sum(axis=1)
	normalized_sum = sum / (mag_base * mag_words)
	return normalized_sum.argmax()

def main():
	parser = argparse.ArgumentParser(description='Given an arbitrary english word, generates an alignment chart of similar words.')
	parser.add_argument(
		'-d', '--dictionary',
		metavar='file',
		default='numberbatch-en-19.08/dictionary.npy.lz4',
		help='A data file containing the dictionary. By default %(default)s.',
	)
	parser.add_argument(
		'-e', '--embeddings',
		metavar='file',
		default='numberbatch-en-19.08/embeddings.npy.lz4',
		help='A data file containing word embeddings of the dictionary. By default %(default)s.',
	)
	parser.add_argument(
		'-n', '--num-neighbors',
		metavar='int',
		type=int,
		default='32',
		help='The number of neighbor words to check for most good, evil, lawful, and chaotic. By default %(default)s.',
	)
	parser.add_argument(
		'words',
		metavar='word',
		nargs='+',
		help='The word(s) to generate alignment charts for.',
	)
	args = parser.parse_args()

	with lz4.frame.open(args.dictionary, 'rb') as f:
		dictionary = np.load(f)
	with lz4.frame.open(args.embeddings, 'rb') as f:
		embeddings = np.load(f)

	with recursionlimit(10000):
		kd_embeddings = scipy.spatial.cKDTree(embeddings, 1000)

	good = embeddings[dictionary == 'good'][0]
	evil = embeddings[dictionary == 'evil'][0]
	lawful = embeddings[dictionary == 'lawful'][0]
	chaotic = embeddings[dictionary == 'chaotic'][0]

	goodness = good - evil
	lawfulness = lawful - chaotic

	most_good = np.abs(goodness).argmax()
	most_lawful = np.abs(lawfulness).argmax()
	alignment_basis = np.identity(embeddings.shape[1])
	alignment_basis[most_good,:] = goodness
	alignment_basis[most_lawful,:] = lawfulness

	base_good = np.zeros(embeddings.shape[1])
	base_good[most_good] = 1

	base_lawful = np.zeros(embeddings.shape[1])
	base_lawful[most_lawful] = 1

	for word_string in args.words:
		factor = 1 / np.sqrt(2)
		try:
			word = embeddings[dictionary == word_string][0]
		except IndexError:
			print(f'Unkown word: {word_string}', file=sys.stderr)
			continue
		_, indices = kd_embeddings.query(word, k=args.num_neighbors + 1)
		neighbors = embeddings[indices]
		neighbors_aligned = neighbors @ np.linalg.inv(alignment_basis)
		word_aligned = neighbors_aligned[0,:]
		neighbors_centered = neighbors_aligned[1:,:] - word_aligned

		word_lawful_good = indices[1 + find_most(base_lawful + base_good, neighbors_centered)]
		word_lawful_neutral = indices[1 + find_most(base_lawful, neighbors_centered)]
		word_lawful_evil = indices[1 + find_most(base_lawful - base_good, neighbors_centered)]
		word_neutral_good = indices[1 + find_most(base_good, neighbors_centered)]
		word_neutral_neutral = indices[0]
		word_neutral_evil = indices[1 + find_most(-base_good, neighbors_centered)]
		word_chaotic_good = indices[1 + find_most(-base_lawful + base_good, neighbors_centered)]
		word_chaotic_neutral = indices[1 + find_most(-base_lawful, neighbors_centered)]
		word_chaotic_evil = indices[1 + find_most(-base_lawful - base_good, neighbors_centered)]

		table = (
			(word_lawful_good,    word_neutral_good,    word_chaotic_good),
			(word_lawful_neutral, word_neutral_neutral, word_chaotic_neutral),
			(word_lawful_evil,    word_neutral_evil,    word_chaotic_evil),
		)

		for row in table:
				print(*dictionary[np.array(row)], sep=' ')

if __name__ == '__main__':
	main()

