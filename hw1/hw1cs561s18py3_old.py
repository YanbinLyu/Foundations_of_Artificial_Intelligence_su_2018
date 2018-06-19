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



# Recursive Functions
# Find all terminal nodes from this situation
def findTerminal(utility, takenRegion, isR1, regionValues, availableRegions, adjacencyMap, maxDepth):
	results = []
	targetUser = 'R1' if isR1 else 'R2'
	nextUser = 'R2' if isR1 else 'R1'
	
	# If there is no available choice for the target user, then we can return directly
	if len(availableRegions[targetUser]) == 0:
		results.append(utility[targetUser])
		return " ", results, utility[nextUser]

	bestChoice = ""
	maxUtil = 0
	maxNextUtil = 0
	availableTargetList = sorted(availableRegions[targetUser], key=regionComp)
	availableNextList = sorted(availableRegions[nextUser], key=regionComp)
	for region in availableTargetList:	
		if maxDepth <= 1:
			results.append(utility[targetUser]+regionValues[region])
			if utility[targetUser]+regionValues[region] > maxUtil and utility[nextUser] >= maxNextUtil:
				maxUtil = utility[targetUser]+regionValues[region]
				bestChoice = region
				maxNextUtil = utility[nextUser]
		else:
			passRound = True
			maxLocalNextUtil = 0
			maxLocalUtil = 0
			for nRegion in availableNextList:
				if nRegion != region:
					passRound = False;
					maxLocalNextUtil = 0
					maxLocalUtil = 0
					utilityCopy = copy.deepcopy(utility)
					utilityCopy[targetUser] += regionValues[region]
					utilityCopy[nextUser] += regionValues[nRegion]
					takenRegionCopy = copy.deepcopy(takenRegion)
					takenRegionCopy[targetUser].add(region)
					takenRegionCopy[nextUser].add(nRegion)
					availableRegionsCopy = copy.deepcopy(availableRegions)
					availableRegionsCopy[targetUser].remove(region)
					if nRegion in availableRegionsCopy[targetUser]:
						availableRegionsCopy[targetUser].remove(nRegion)
					availableRegionsCopy[nextUser].remove(nRegion)
					if region in availableRegionsCopy[nextUser]:
						availableRegionsCopy[nextUser].remove(region)
					for r1Region in adjacencyMap[region]:
						if r1Region not in takenRegionCopy['R1'] and r1Region not in takenRegionCopy['R2']:
							availableRegionsCopy[targetUser].add(r1Region)
					for r2Region in adjacencyMap[nRegion]:
						if r2Region not in takenRegionCopy['R1'] and r2Region not in takenRegionCopy['R2']:
							availableRegionsCopy[nextUser].add(r2Region)
					maxDepthCopy = maxDepth - 2
					nChoice, nextResults, nUtility = findTerminal(utilityCopy, takenRegionCopy, isR1, regionValues, availableRegionsCopy, adjacencyMap, maxDepthCopy)
					if maxLocalUtil < max(nextResults):
						maxLocalUtil = max(nextResults)
						if maxLocalNextUtil <= nUtility:
							maxLocalNextUtil = nUtility
					results = results + nextResults

			if passRound:
				utilityCopy = copy.deepcopy(utility)
				utilityCopy[targetUser] += regionValues[region]
				takenRegionCopy = copy.deepcopy(takenRegion)
				takenRegionCopy[targetUser].add(region)
				availableRegionsCopy = copy.deepcopy(availableRegions)
				availableRegionsCopy[targetUser].remove(region)
				if region in availableRegionsCopy[nextUser]:
					availableRegionsCopy[nextUser].remove(region)
				for r1Region in adjacencyMap[region]:
					if r1Region not in takenRegionCopy['R1'] and r1Region not in takenRegionCopy['R2']:
						availableRegionsCopy[targetUser].add(r1Region)
				maxDepthCopy = maxDepth-2
				nChoice, nextResults, nUtility = findTerminal(utilityCopy, takenRegionCopy, isR1, regionValues, availableRegionsCopy, adjacencyMap, maxDepthCopy)
				if maxLocalUtil < max(nextResults):
					maxLocalUtil = max(nextResults)
					if maxLocalNextUtil <= nUtility:
						maxLocalNextUtil = nUtility
				results = results + nextResults
			if maxLocalUtil > maxUtil:
				bestChoice = region
				maxUtil = maxLocalUtil
				maxNextUtil = maxLocalNextUtil
	return bestChoice, results, maxNextUtil


####################### Main Function ########################################
# Read input
# Dictionary adjacencyMap: the dictionary containing the adjacent regions of each region
adjacencyMap = {}
# List regions: the name of the regions based on alphabet order
regions = []
regionOrders = {}
index = 0
f1 = open(sys.argv[1])
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
print(utility)
print()
print(takenRegion)
print()
print(regionValues)
print()
print(availableRegions)
print()
print(adjacencyMap)
print()
bestChoice, results, maxNextUtil = findTerminal(utility, takenRegion, isR1, regionValues, availableRegions, adjacencyMap, maxDepth)
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