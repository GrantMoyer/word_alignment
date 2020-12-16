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

def main():
	parser = argparse.ArgumentParser(description='Given an arbitrary english word, generates an alignment chart of similar words.')
	parser.add_argument(
		'-w', '--words',
		metavar='file',
		default='numberbatch-en-19.08-words.npy.lz4',
		help='A data file containing word embeddings. By default %(default)s.',
	)
	parser.add_argument(
		'-e', '--embeddings',
		metavar='file',
		default='numberbatch-en-19.08-embeddings.npy.lz4',
		help='A data file containing word embeddings. By default %(default)s.',
	)
	args = parser.parse_args()

	with lz4.frame.open(args.words, 'rb') as f:
		words = np.load(f)
	with lz4.frame.open(args.embeddings, 'rb') as f:
		embeddings = np.load(f)

	with recursionlimit(10000):
		kd_embeddings = scipy.spatial.cKDTree(embeddings, 1000)

if __name__ == '__main__':
	main()

