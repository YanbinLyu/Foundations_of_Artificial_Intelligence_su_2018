import sys
import string
import re
import copy
import math

# Compare function, use the order of appearance since the regions will be given in alphabetical order predefined. 
# For Python 3, I changed the compare function to key function
def regionComp(x):
	return regionOrders[x]

def utilComp(x):
	return regionValues[x]

infinity = float('inf')


# Recursive Functions
# Find all terminal nodes from this situation
def findMaxMax(isR1, isTarget, utility, takenRegion, availableRegions, maxDepth):
	results = []
	currUser = 'R1' if isR1 else 'R2'
	nextUser = 'R2' if isR1 else 'R1'
	availableCurrentList = sorted(availableRegions[currUser], key=regionComp)
	maxUtil = -infinity
	maxUtilities = {}
	bestChoice = "Yes"
	if len(availableCurrentList) == 0:
		
		if isTarget:
			results.append(utility[currUser])
		else:
			nUtility, nChoice, nResults = findMaxMax(not isR1, not isTarget, utility, takenRegion, availableRegions, maxDepth-1)
			results = results + nResults
		return utility, bestChoice, results
	
	for region in availableCurrentList:
		if maxDepth <= 0:
			maxUtilities[currUser] = utility[currUser]
			maxUtilities[currUser] += regionValues[region]
			maxUtilities[nextUser] = utility[nextUser]
			if isTarget:
				results.append(maxUtilities[currUser])
				bestChoice = region
			else:
				results.append(maxUtilities[nextUser])
				bestChoice = region	
		else:
			utilityCopy = copy.deepcopy(utility)
			utilityCopy[currUser] += regionValues[region]
			takenRegionCopy = copy.deepcopy(takenRegion)
			takenRegionCopy[currUser].add(region)
			availableRegionsCopy = copy.deepcopy(availableRegions)
			if region in availableRegionsCopy[nextUser]:
				availableRegionsCopy[nextUser].remove(region)
			availableRegionsCopy[currUser].remove(region)
			for adjacentRegion in adjacencyMap[region]:
				if adjacentRegion not in takenRegionCopy['R1'] and adjacentRegion not in takenRegionCopy['R2']:
					availableRegionsCopy[currUser].add(adjacentRegion)
			nUtility, nChoice, nResults = findMaxMax(not isR1, not isTarget, utilityCopy, takenRegionCopy, availableRegionsCopy, maxDepth-1)
			
			results = results + nResults
			if nUtility[currUser] > maxUtil:
				maxUtil = nUtility[currUser]
				maxUtilities[currUser] = nUtility[currUser]
				maxUtilities[nextUser] = nUtility[nextUser]
				bestChoice = region
	return maxUtilities, bestChoice, results
				

# Alpha Beta Alg
def minimax(isR1, isTarget, alpha, beta, availableRegions, takenRegion, maxDepth, utilityValue):
	currUser = 'R1' if isR1 else 'R2'
	nextUser = 'R2' if isR1 else 'R1'
	results = []
	bestChoice = ""
	availableCurrentList = sorted(availableRegions[currUser], key=regionComp)
	if maxDepth < 0:
		results.append(utilityValue)
		return utilityValue, results, bestChoice
	if len(availableCurrentList) == 0:
		if isTarget:
			results.append(utilityValue)
			return utilityValue, results, bestChoice
		else:
			value, nResults, nChoice = minimax(not isR1, True, alpha, beta, availableRegions, takenRegion, maxDepth-1, utilityValue)
		return value, nResults, nChoice
	
	if isTarget:
		bestVal = -infinity
		for region in availableCurrentList:
			utilityValueCopy = utilityValue + regionValues[region]
			takenRegionCopy = copy.deepcopy(takenRegion)
			takenRegionCopy[currUser].add(region)
			availableRegionsCopy = copy.deepcopy(availableRegions)
			if region in availableRegionsCopy[nextUser]:
				availableRegionsCopy[nextUser].remove(region)
			availableRegionsCopy[currUser].remove(region)
			for adjacentRegion in adjacencyMap[region]:
				if adjacentRegion not in takenRegionCopy['R1'] and adjacentRegion not in takenRegionCopy['R2']:
					availableRegionsCopy[currUser].add(adjacentRegion)
			value, nResults, nChoice= minimax(not isR1, False, alpha, beta, availableRegionsCopy, takenRegionCopy, maxDepth-1, utilityValueCopy)
			if value > bestVal:
				bestChoice = region
			results = results + nResults
			bestVal = max(bestVal, value)
			alpha = max(alpha, bestVal)
			if beta <= alpha:
				break
		return bestVal, results, bestChoice
	else:
		bestVal = infinity
		for region in availableCurrentList:
			utilityValueCopy = utilityValue
			takenRegionCopy = copy.deepcopy(takenRegion)
			takenRegionCopy[currUser].add(region)
			availableRegionsCopy = copy.deepcopy(availableRegions)
			if region in availableRegionsCopy[nextUser]:
				availableRegionsCopy[nextUser].remove(region)
			availableRegionsCopy[currUser].remove(region)
			for adjacentRegion in adjacencyMap[region]:
				if adjacentRegion not in takenRegionCopy['R1'] and adjacentRegion not in takenRegionCopy['R2']:
					availableRegionsCopy[currUser].add(adjacentRegion)
			value, nResults, nChoice = minimax(not isR1, True, alpha, beta, availableRegionsCopy, takenRegionCopy, maxDepth-1, utilityValueCopy)
			if value > bestVal:
				bestChoice = region
			results = results + nResults
			bestVal = min(bestVal, value)
			beta = min(beta, bestVal)
			if beta <= alpha:
				break
		return bestVal, results, bestChoice


####################### Main Function ########################################
# Read input
# Dictionary adjacencyMap: the dictionary containing the adjacent regions of each region
adjacencyMap = {}
# List regions: the name of the regions based on alphabet order
regions = []
regionOrders = {}
index = 0
f1 = open(sys.argv[2])
# isToday: whether the data is updated today
if f1.readline().strip('\n') == "Today":
	isToday = True
else:
	isToday = False
# isR1: whether the next player is R1
if f1.readline().strip('\n') == "R1":
	isR1 = True
else:
	isR1 = False
# Dictionary regionValues: the dictionary of the values of different regions
regionValues = {}
valueTuples = re.findall(r"[\w']+", f1.readline().strip('\n'))
pos = 0
while pos < len(valueTuples)/2:
	regionName = valueTuples[pos*2]
	regionValues[regionName] = int(valueTuples[pos*2+1])
	adjacencyMap[regionName] = []
	regions.append(regionName)
	regionOrders[regionName] = index
	index += 1
	pos += 1
# numRegions: Number of reigions
numRegions = pos
# Update heuristic values
if not isToday:
	total = 0
	for key, value in regionValues.items():
		total += value
	for key, value in regionValues.items():
		regionValues[key] = (value + total/numRegions)/2
# regions = sorted(regions) Since the professor specify that the RPL will be given in alphabetical order
for i in range(0, numRegions):
	row = f1.readline().strip('\n')[1:-1].split(',')
	for j in range(0, numRegions):
		if row[j] == '1' and i != j:
			adjacencyMap[regions[i]].append(regions[j])
# Dictionary takenRegion: the dictionary that R1 and R2 have already picked
takenRegion = {}
takenRegion['R1'] = set()
takenRegion['R2'] = set()
# Dictionary utility: the dictionary that R1 and R2 have already got from the picked so far
utility = {}
utility['R1'] = 0
utility['R2'] = 0
picked = f1.readline().strip('\n').split(',')
currR1 = isR1
if picked[0] != '*':
	for i in range(0, len(picked)):
		nextRegion = picked[len(picked)-1-i] 
		if nextRegion != 'PASS':
			takenRegion['R2' if currR1 else 'R1'].add(nextRegion)
			utility['R2' if currR1 else 'R1'] += regionValues[nextRegion]
		currR1 = False if currR1 else True
# int maxDepth: the maximum depth of my search
maxDepth = int(f1.readline().strip('\n'))
if picked[0] != '*':
	maxDepth = maxDepth - len(picked)
f1.close()

availableRegions = {}
availableRegions['R1'] = set()
availableRegions['R2'] = set()
if len(takenRegion['R1']) > 0:
	for region in takenRegion['R1']:
		for nRegion in adjacencyMap[region]:
			if nRegion not in takenRegion['R1'] and nRegion not in takenRegion['R2']:
				availableRegions['R1'].add(nRegion)
else:
	for region in regions:
		if region not in takenRegion['R2']:
			for nRegion in adjacencyMap[region]:
				if nRegion not in takenRegion['R2']:
					availableRegions['R1'].add(nRegion)
if len(takenRegion['R2']) > 0:
	for region in takenRegion['R2']:
		for nRegion in adjacencyMap[region]:
			if nRegion not in takenRegion['R2'] and nRegion not in takenRegion['R1']:
				availableRegions['R2'].add(nRegion)
else:
	for region in regions:
		if region not in takenRegion['R1']:
			for nRegion in adjacencyMap[region]:
				if nRegion not in takenRegion['R1']:
					availableRegions['R2'].add(nRegion)
# maxUtilities, bestChoice, results = findMaxMax(isR1, True, utility, takenRegion, availableRegions, maxDepth)
bestValue, results, bestChoice = minimax(isR1, True, -infinity, infinity, availableRegions, takenRegion, maxDepth, utility['R1' if isR1 else 'R2'])
# Output file
f2 = open('output.txt', "w+")
f2.write(bestChoice)
f2.write("\n")
for num in range(0, len(results)-1):
	f2.write(str(int(results[num]+0.5)))
	# f2.write(str(results[num]))
	f2.write(",")
f2.write(str(int(results[len(results)-1]+0.5)))
f2.close()