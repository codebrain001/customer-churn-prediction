# **Customer Churn Prediction**

Churn rate is an efficient indicator for subscription-based companies. Identifying customers who aren't happy with provided solutions allows businesses to learn about product or pricing plan weak points, operation issues, and customer preferences and expectations to reduce reasons for churn proactively.

Designing churn prediction workflow
The overall scope to build an ML-powered application to forecast customer attrition is generic to standardized ML project structure that includes the following steps:

1. Defining the problem and the goal: It's essential to understand what insights one needs to get from the analysis and prediction. Understanding the problem and gathering requirements of stakeholders' pain points and expectations.
2. Establishing data source: Next, specifying data sources is necessary for the following stage of modeling. Some popular sources of churn data are CRM systems, Analytics services, and customer feedback.
3. Data preparation, exploration, and preprocessing: Selected raw historical data for solving the problem and building predictive models requires transformation into a format suitable for machine learning algorithms. This step can also aid improvement of the overall results due to an increase in the quality of data.
4. Modeling and testing: This covers the development and performance validation of the customers' churn prediction models with various machine learning algorithms.
5. Deployment and monitoring: This is the last stage in the life cycle of the development of machine learning for churn rate forecasts. Here, the most suitable model is sent into production. It can be either integrated into existing software or become a core for a newly built application.


The application to be deployed will function via operational use cases:
Online prediction: This use case generates predictions on a one-by-one basis for each data point (in the context of this article, a customer).
Batch prediction: This use is for generating predictions for a set of observations instantaneously.

![Alt Text](streamlit-app.gif)
