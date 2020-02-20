using CSV

using CPLEX
using DataFrames
using Random
using Pkg

Pkg.activate(pwd())
Pkg.add(PackageSpec(name="JuMP", version="0.18.6"))
#Pkg.add(PackageSpec(name="vOptGeneric"))
using JuMP
using vOptGeneric
using Statistics

include("functionsMultiObj.jl")

dataSet = "multiclass_multiobj" # "multiclass_multiobj" or "multiobj_kidney"
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
rules = createRules(dataSet, resultsFolder, train)

# Order the rules (limit the resolution to 300 seconds)
# Details:
# - read the file ./data/kidney_rules.csv
# - save the rules in ./res/kidney_ordered_rules.csv
timeLimitInSeconds = 600
orderedRules = sortRules(dataSet, resultsFolder, train, rules, timeLimitInSeconds)

println("-- Train results")
showStatistics(orderedRules, train)

println("-- Test results")
showStatistics(orderedRules, test)
