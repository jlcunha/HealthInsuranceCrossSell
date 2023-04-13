# EverGuardian

![alt text](img/everguardian.png)


**This project is a research study with the objective of calculating the propensity of customers to purchase vehicle insurance, using data from existing customers who have health insurance. By targeting only the customers most likely to purchase, the company can reduce the cost of unnecessary calls. The resulting cost savings make the operation 72% more profitable.**

---

## 1 - Business Problems

### 1.1 - Ever Guardian

Ever-Guardian is an insurance company that provides health insurance to its customers. With the objective of **maximizing revenue**, Ever-Guardian is starting to **sell vehicle insurance**. Initially, in an effort to **reduce the Customer Acquisition Cost (CAC)**, the insurance company adopted a **cross-selling** strategy to sell a second product to existing customers.

Cross-selling is a sales strategy in which a company offers existing customers the opportunity to purchase additional products or services. This strategy aims to increase revenue by encouraging customers to buy more than one product from the same company. By cross-selling, companies can **reduce customer acquisition costs and increase customer loyalty**, as customers are more likely to remain with a company that offers a range of products and services that meet their needs. In the case of Ever Guardian, adopting a cross-selling strategy by selling vehicle insurance to their existing health insurance customers can help the company **reduce customer acquisition costs and increase revenue**.

### 1.2 - Business Problems

For a more effective campaign, Ever-Guardian Company has decided to recommend car insurance **only to the customers who are most likely to be interested in it**. This will enable the call center to **make targeted calls to potential customers** who are more likely to purchase car insurance. As a data scientist at Ever-Guardian, my role is to develop a predictive model to determine a customer's propensity to buy vehicle insurance.


### 1.3 - About the data

---
| Column | Description |
| --- | --- |
|id|                    Unique ID for each customer |
|gender|                Customer gender |
|age|                   Customer age |
|region_code|           Region code where the customer lives |
|policy_sales_channel|  Channel code of contact chosen by the customer |
|driving_license|       Does the customer have a driving license? |
|vehicle_age|           Age of the customer's car |
|vehicle_damage|        Was the customer's car damaged? |
|previously_insured|    Has the customer been insured previously? |
|annual_premium|        Annual premium value (for health insurance) |
|vintage|               Number of days for which the customer has health insurance |
|response|              Is the customer interested in having car insurance? |




---



## 2 - Solution Strategy

**CRISP-DM** stands for Cross Industry Standard Process for Data Mining, which can be translated as Standard Process for Inter-Industry Data Mining. It is a data mining process model that describes commonly used approaches by data mining experts to tackle problems.

<img src="img/crisp.jpg" width="500">

As a basis for this project, I use CRISP, and now, in the second cycle of CRISP, I made further iterations, create new features, generate new insights, and improve model performance, all to deliver more value to the company. It is also important to engage with stakeholders in the project throughout the process, to keep them informed and manage their expectations.


All code, visualizations, and analyses for this project can be found at the following links:

[Cycle 02](notebook/v03.ipynb)

[Cycle 01](notebook/v03.ipynb)

### 2.1 - Customer Surveys

The first step in this project is to select a sample of customers and survey them to determine their interest in the new product. Using this data, we can develop a model to predict each customer's propensity to buy car insurance.

In this project, we already have the response data from interested customers, which is ready to be used to develop the model. But for it, I have to collect the data in a SQL database.

<img src="img/query.png" width="500">



### 2.2 - Collecting data into an SQL database

To retrieve the data, I will connect to the SQL database, specifically the PostgreSQL database used by the insurance company. I will then use a Python library that can convert SQL queries into pandas dataframes, allowing for easy manipulation and analysis of the data.

## 3 - Next Steps
1. A sales forecast using the Facebook Prophet framework.

2. Seasonality analysis.

3. Increase customer data collection.
Fetch new data from customer regions.

4. From the increase in the number of data, calculate the price elasticity.

## 4 - Technologies

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)](https://scipy.org/)
[![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com/)

## 5 - Author

Lucas da Cunha

Data Scientist / Data Analyst

[Project Portfolio](https://jlcunha.github.io/portfolio_projetos/)

[GitHub Profile](https://github.com/jlcunha/)
