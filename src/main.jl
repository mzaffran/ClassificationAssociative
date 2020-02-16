using CSV
using JuMP
using CPLEX
using DataFrames
using Random
using Statistics

include("functions.jl")

dataSet = "multiclass"
dataFolder = "./data/"
resultsFolder = "./res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
# Details:
# - read the file ./data/kidney.csv
# - save the features in ./data/kidney_test.csv and ./data/kidney_train.csv
train, test = createFeatures(dataFolder, dataSet)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
# Details:
# - read the file ./data/kidney_train.csv
# - save the rules in ./res/kidney_rules.csv
timeLimitInSecondsCreate = 300
rules = createRules(dataSet, resultsFolder, train, timeLimitInSecondsCreate)

# Order the rules (limit the resolution to 300 seconds)
# Details:
# - read the file ./data/kidney_rules.csv
# - save the rules in ./res/kidney_ordered_rules.csv
timeLimitInSecondsSort = 300
orderedRules = sortRules(dataSet, resultsFolder, train, rules, timeLimitInSecondsSort)

println("-- Train results")
showStatistics(orderedRules, train)

println("-- Test results")
showStatistics(orderedRules, test)
