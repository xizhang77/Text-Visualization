
def Accuracy(result, labels):
	n = len(labels)
	score = 0.0
	for i in range(n-1):
		for j in range(i+1, n):
			score = score + ( (labels[i] == labels[j]) == (result[i] == result[j]))

	return score/(0.5 * n * (n-1))