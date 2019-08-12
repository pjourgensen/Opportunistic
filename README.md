# NHANES Analysis

#### -- Project Status: [On Hold]

## Intro/Objective
The purpose of this project is to develop a cancer classification and risk prediction model in the hopes of providing advance warning to patients at risk for cancer.  Every year, the National Health and Nurtitional Examination Survey (NHANES) gathers patient data detailing symptoms, demographic information, family history, diet, lab results, and comorbidities (among other considerations). This data is made available to allow the general public to analyze and develop models based on the responses. The main difficulty lies within the quality of the data as many of the responses are missing. In this project, I present an analysis of feature selection and engineering before developing an object-oriented pipeline that allows for seamless trials of parameterizations. Ultimately, a gradient-boosted decision tree model performed optimally, achieving a 24.2% recall on a heldout test set. I believe there is still room for improvement in the feature selection and preprocessing steps, so this project remains [On Hold] until the opportunity to continue arises.

### Methods Used
* Object-Oriented Programming
* Pipelining
* Inferential Statistics
* Missing Value Imputation, Categorical Encoding
* Feature Selection, Filtering
* Decision Trees, SVMs
* Ensemble Methods, Boosting
* Data Visualization
* Predictive Modeling
* Model Persistance

### Technologies
* Python, jupyter
* Pandas, Numpy
* SKLearn
* Tensorflow
* Seaborn, Matplotlib
* joblib

## Project Description
* Data  
   * 50,000 patients with over 500 possible survey responses
   * Roughly 9% are positive for having cancer at some point in their lives
   * Requires substantial hand-selection and preprocessing
* Feature Selection
   * Began with a correlational filter to remove highly correlated features 
   * Passed data through mutual information filter to identify subset with some relationship to the target class
   * Finally passed data through variance filter to remove features with minimal variance
   * Lasso regression and tree-based methods were also tested, but resulted in declines in model performance
* Performance Metric
   * Chose 5-fold cross-validation on balanced training set to assess overall accuracy
* Predictive Modeling
   * Ran parameter tuning grid searches over Decision Trees, SVMs, Random Forests, GBDTs, and Neural Networks
   * Found that an optimized GBDT delivered an accuracy of 76.5% and recall of 24.2%

## Getting Started

1. Clone this repo.
2. Download Raw Data from [here](https://drive.google.com/file/d/1hFp7O747408D8t5442f0Sjit7wXKXI1z/view?usp=sharing).    
3. Download the requirements.
4. Edit sample_config to your preferences.
5. Edit config_outpath, config_inpath, and joblib_outpath with NhanesModelManager
6. Run NhanesModelManager

## For more detail and discussion:
* [Blog Post](https://pjourgensen.github.io/nhanes.html)

