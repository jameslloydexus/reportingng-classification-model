## Reporting NG Classification model

This repo contains the code to fit the various logistic regression models for the reporting NG classification problems. 

The data for the models are the result of the data processing pipeline that the batch job executes. 

The problems are two fold: the Right Person Contacted (RPC) classification problem, and the Promise to Pay (PtP) classification problem. These problems are further split between product and bucket (between consumer loans and credit cards and between buckets 2 and 3). As a result, there will be 8 classification models in total. 

