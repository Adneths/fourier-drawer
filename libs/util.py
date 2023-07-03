def printProgressBar(percentage, prefix = '', suffix='', decimals = 1, length = 40):
	percent = ("{0:." + str(decimals) + "f}").format(100*percentage)
	fill = int(length * percentage)
	bar = '*' * fill + '-' * (length - fill)
	print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = '')