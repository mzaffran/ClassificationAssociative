using CSV
using JuMP
using CPLEX
using DataFrames
using Random
using Statistics


include("functions.jl")

dataSet = "kidney"
dataFolder = "./data/"
resultsFolder = "./res/"

# Create the features tables (or load them if they already exist)
# Note: each line corresponds to an individual, the 1st column of each table contain the class
# Details:
# - read the file ./data/kidney.csv
# - save the features in ./data/kidney_test.csv and ./data/kidney_train.csv
deleteData = false
respectProp = true
train, test = createFeatures(dataFolder, dataSet, deleteData, respectProp)

# Create the rules (or load them if they already exist)
# Note: each line corresponds to a rule, the first column corresponds to the class
# Details:
# - read the file ./data/kidney_train.csv
# - save the rules in ./res/kidney_rules.csv
timeLimitInSecondsCreate = 600
deleteCreate = false
t1 = time_ns()
rules = createRules(dataSet, resultsFolder, train, timeLimitInSecondsCreate, deleteCreate)
t2 = time_ns()
if deleteCreate == true
    println("-- Elapsed time to generate rules ",(t2-t1)/1.0e9, "s")
end

# Order the rules (limit the resolution to 300 seconds)
# Details:
# - read the file ./data/kidney_rules.csv
# - save the rules in ./res/kidney_ordered_rules.csv
timeLimitInSecondsSort = 600
deleteSort = false
t1 = time_ns()
orderedRules = sortRules(dataSet, resultsFolder, train, rules, timeLimitInSecondsSort, deleteSort)
t2 = time_ns()
if deleteSort == true
    println("-- Elapsed time to order rules ",(t2-t1)/1.0e9, "s")
end

println("=== Results")

println("-- Train results")
showStatistics(orderedRules, train)

println("-- Test results")
showStatistics(orderedRules, test)
