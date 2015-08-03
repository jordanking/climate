# Climate
A project to find a quick way to classify temperature time series readings as originating from one of three top climate models.

# Purpose
Analyze the accuracy to which we can identify which climate model a time series of temperature readings comes from.
This is a challenge used to research bias in climate models, done to assist Lawrence Livermore National Lab.

# Method
Nearest neighbor methods tend to work well as models have strong biases, 
though speed and preprocessing become big players when aiming for perfect classification. 
We average training data from each run of each model, producing 3 time series to compare against.
These time series are rounded to the nearest 2.5 degrees lat/long as the test data is (an artifact of the challenge).
Geographic nearest neighbors are found next within a 4 degree lat/long radius.
The nearest time series within the ball to the test point is used to define the classification model.
