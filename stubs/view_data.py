#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():

	df = pd.read_csv('/Users/parthpendurkar/Desktop/raw.csv')
	print('{} rows...'.format(len(df)))

	unacceptable = df[df['Classification'] == 1]
	acceptable = df[df['Classification'] == 0]

	print('{} classified as unacceptable...'.format(len(unacceptable)))
	print('{} classified as acceptable...'.format(len(acceptable)))

	from pdb import set_trace
	set_trace()


if __name__ == '__main__':
	main()