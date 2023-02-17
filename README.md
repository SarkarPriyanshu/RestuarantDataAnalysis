# Restaurant Data Analysis

## Summary | Problem Statement

### Summary :
Machine Learning Project where we done analysis on Zomato dataset using historic data.This model wil predict the estimated cost for two people based on various factor of curtain Restaurants. 

### About Dataset :
 * [Zomato Dataset](https://www.kaggle.com/datasets/pranavuikey/zomato-eda)
    - Data Cleaning:
        - Deleting redundant columns.
        - Renaming the columns.
        - Dropping duplicates.
        - Cleaning individual columns.
        - Remove the NaN values from the dataset
        - Check for some more Transformations
    - Data Columns:
        - Restaurants delivering Online or not
        - Restaurants allowing table booking or not
        - Table booking Rate vs Rate
        - Best Location
        - Relation between Location and Rating
        - Restaurant Type
        - Gaussian Rest type and Rating
        - Types of Services
        - Relation between Type and Rating
        - Cost of Restaurant
        - No. of restaurants in a Location
        - Restaurant type
  
     
### Problem Statement :  
We are trying to use a restaurant data set to predict the ratings of two people based on different characteristics of the restaurant. The dataset contains information about the city, location, type of cuisine served, price, rating, and more. The goal is to use this data to develop a model that can accurately predict two people's ratings  based on restaurant characteristics. This can be done by creating a regression model that uses restaurant characteristics to predict the rating The model can then be used to predict two people's rating  for a particular restaurant

## Table of Contents
  - [Basic Analysis](#Analysis-of-Data)
  - [Feature Engineering Analysis](#feature-engineering-analysis)
  - [Model Selection](#model-selection-and-evaluations)
  - [Feature Selection](#feature-selection)    
  - [Model Selection and Evaluations](#model-selection-and-evaluations)
  - [Custom Classes & Optimisation](#creating-custom-pipeline-classes-and-optimisation)

## Basic Analysis
### Analysis of Data
 
>>
    * Univarient Analysis for Numerica Variables.
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis1.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis2.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis3.png?raw=true)
    
    * Univarient Analysis for Categoral variables
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis4.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis5.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/UnivariantAnalysis6.png?raw=true)

    * Observations:
        - Null value Handling
        - Skewed data , feature Tranformation
        - Outliers handling
        - handling high cardinality
        - curce of dimentionality handling






## Feature Engineering Analysis
###  Note Preprocesing and cleaning:

>  
    * Null value handling
        - Observations:
            - replace rate variable '-' with nan value and then apply mode on it 
            - approx_cost(for two people) replace ',' type conversion to float
            - menu_item replace [] into np.nan
            - location, cuisines, rest_type mode imputation for missing less than 5% data missing.
            - rate, dish_liked, menu_item missing indicator for missing data as more than 5% data is missing
            - votes have 0 in it convert then to nan and due to missing value is more than 17% we add a column for missing indication convert type to int from float.
            - Need to preprocess menu_item column throughly  

### Observation after preprocessing
>>
    * Feature Transformaions
        - Observation: 
            - We have applied log transformation on target columns based on the above observation we can see that it converge into gaussion distribution so we can apply log transformation on dependent numeric feature.
            
            - Log tranfoemation for votes as our target column converge to noral distribution and also handled outliers well
        
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/Logerithmic%20Transformation.png?raw=true)

![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/Logerithmic%20Transformation1.png?raw=true)
    
    
    
    * handling high cardinality
        - Observations: 
            online_order 2
            book_table 2
            rate 63
            location 93
            rest_type 93
            dish_liked 5271
            cuisines 2723
            menu_item 9097
            listed_in(type) 7
            listed_in(city) 30
            
            - rates convert string to float and then convert them to categorical variable.
            - Groupby location see relation with target column based on that select top 10 categories and rest to Rare label and then label them to numeric
            - online_order and book_table replace Yes to 1 and No to 0
    
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/ratevstarget.png?raw=true)
        
        - Rate have a monotic relation with target column we convert rate into categorical feature and then based on monotic realation convert rate feature using ordinal encoding.

![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/locationvstarget.png?raw=true)

    - Same relation we find with locations and other two feature what we did is we take top 10 values have relation with target column and rest convert them to rare category.
    
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/listinvs%20target.png?raw=true)    


![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/listedinvstarget2.png?raw=true)  

    - Same relation we find with listin and other two feature what we did is we take top 10 values have relation with target column and rest convert them to rare category.

![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/listedincityvstarget.png?raw=true)

![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/listedincityvstarget1.png?raw=true)   

    - Same relation we find with listin(City) and other two feature what we did is we take top 10 values have relation with target column and rest convert them to rare category.
    
![foxdemo](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/Graphs/resttypevstarget.png?raw=true) 

    - Same relation we find with rest-type and cusine and other two feature what we did is we take top 10 values have relation with target column and rest convert them to rare category.

## Feature Selection
> ### List of selected columns 
        'online_order', 'book_table', 'rate', 'votes', 'location',
       'listed_in(type)', 'listed_in(city)', 'rest_type_Fine Dining',
       'rest_type_Bar', 'rest_type_Microbrewery', 'rest_type_Lounge',
       'rest_type_Club', 'rest_type_Casual Dining', 'rest_type_Quick Bites',
       'rest_type_Pub', 'rest_type_Cafe', 'rest_type_Irani Cafee',
       'rest_type_Food Court', 'rest_type_Dessert Parlor', 'rest_type_Bakery',
       'rest_type_Rare', 'cuisines_Bar', 'cuisines_Cafe', 'cuisines_Bakery',
       'cuisines_Rare'
       
       total features: 44
       selected features: 25
       features with coefficients shrank to zero: 19
    

## Model Selection and Evaluations
> ### DecisionTreeRegressor and evaluation
>>
     Observation DecisionTreeRegressor:
    
    * R2 Score: 0.7437300782349615
    * Test R2 Score: 0.7045291282847792
    * Best Parameters: 'max_depth': 8,
                       'max_leaf_nodes': 14,
                       'min_samples_leaf': 3,
                       'min_samples_split': 8,
                       'min_weight_fraction_leaf': 0.0
        
> ### RandomForestRegressor and evaluation
>>
     Observation Support RandomForestRegressor:
     
    *  Train R2 Score: 0.8515083816801694
    *  Test R2 Score: 0.836995164152063
    *  Best Parameters: 'bootstrap': True,
                            'max_depth': 10,
                            'min_samples_leaf': 2,
                            'min_samples_split': 2,
                            'n_estimators': 100,
                            'oob_score': True
            

  
## Creating Custom Pipeline Classes and Optimisation.
> ### [Custom Pipeline](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/CustomerBehaviorpipeline.ipynb)
>>
    * Drawback Execution Time 
        The time of execution of above program is : 700138.5018825531 ms

> ### [Optimised Custom Pipeline](https://github.com/SarkarPriyanshu/RestuarantDataAnalysis/blob/main/OptimizedCustomerBehavior.ipynb)
    * Advantage Execution Time
        The time of execution of above program is : 472.80359268188477 ms

### [Sponsor By me]([https://github.com/SarkarPriyanshu])

## Technologies Used

- [Jupyter Notebook](https://jupyter.org/), for Cleaning, Model traning & Evaluation

## Chao

A passion project by Priyanshu Sarkar here is my [Github](https://github.com/SarkarPriyanshu) and  [CodeSandBox](https://codesandbox.io/u/SarkarPriyanshu)
