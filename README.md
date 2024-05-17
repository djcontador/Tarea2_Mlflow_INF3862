# Property-Friends Real State case

This project leverages machine learning techniques to analyze and understand patterns within real estate data. The model is trained using a diverse dataset containing relevant property features and corresponding valuations. The trained model is then integrated into a scalable and deployable solution, ensuring ease of use and accessibility for the end-users.

The motivation behind this project stems from the client's need for a reliable and efficient property valuation system. By automating the valuation process, the client can streamline their operations, make more informed decisions, and enhance overall business efficiency. The model's accuracy and the deployment-ready solution contribute to the project's significance in addressing the client's specific requirements.

The project delivers a quick deployable solution that is ready for production use. This solution is designed to minimize the time and effort required for implementation, allowing the client to integrate the property valuation model seamlessly into their existing infrastructure.

The primary deliverables of this project include:

1. **Robust Pipeline:** A well-defined and automated pipeline for model training, evaluation, and deployment.

2. **API Development:** An API that can efficiently receive property information and provide accurate valuation predictions in real-time.

3. **Scalable Solution:** A deployable solution that is scalable, ensuring its adaptability to varying workloads and data volumes.


## Dependencies

- Make sure you have all the necessary dependencies installed. You can use the following command to install them:
`pip install -r requirements.txt`

- Also both the Pipeline and API should run using Docker, a platform to automate the deployment of the model inside containers for easy package & distribution of the applications.


### Directory Structure

<pre>
project_root/
|-- README.md
|-- config.py
|-- data/
|   |-- 
|   |-- 
|-- src/
|   |-- __init__.py
|   |-- database.py
|   |-- model/
|       |-- __init__.py
|       |-- train_model.py
|       |-- evaluate_model.py
|       |-- predict_model.py
|-- api/
|   |-- __init__.py
|   |-- api.py
|   |-- Dockerfile
|   |-- requirements.txt
|-- Dockerfile
|-- requirements.txt
|-- pipeline.py
|-- property_valuation_pipeline.joblib
|-- api_logs.log
</pre>

### Module Responsabilities

- pipeline.py: Represents the main entry point for the application
- src/train_model.py: Contains the code for loading data, preprocessing and training the machine learning model.
- src/evaluate_model.py: Contains the code for evaluating the trained model.
- src/predict_model.py: Contains the code for making predictions using the trained model.
- data directory: Contains the training and testing datasets.
- src/database.py: Contains the abstraction for client databases connection to te pipeline directly. 
- api/api.py: Contains the FastAPI application code for serving predictions.

## 1. Pipeline for automatic trainning, evaluation & deployment

### Instructions to Run "The Pipeline App"

**1.1 Data availability**
- In the data directory, import the datasets `train.csv`, `test.csv`. This datasets are not available in this repository due is property of the client

**1.2 Running the main script**
- run the main script from the project root directory through the command line or terminal `python pipeline.py`
- This script will execute the entire pipeline, including loading data, training the model, testing it and evaluating the performance metrics.

**1.3 Run the Docker Container**
In the terminal or in Docker Playground use the command `docker run -dp 0.0.0.0:3000:3000 djcontadorz/the-pipeline-app` for running the Pipeline docker. 

**1.4 Accessing the Application**
Once the container is running, you can access "The Pipeline App" through your browser or API client at http://localhost:3000


#### Prediction Metrics

##### RMSE: Root Mean Squared Error
- Definition: RMSE is the square root of the mean of the squared differences between the predicted values and the actual (observed) values. It measures the standard deviation of prediction errors or residuals.
- Interpretation: RMSE will give you an idea of how much error there is between the predicted property values and the actual values. A lower RMSE is better, as it indicates less deviation from the actual values. However, RMSE can be sensitive to outliers and may overemphasize large errors.

##### MAPE: Mean Absolute Percentage Error
- Definition: MAPE measures the average magnitude of the errors in a set of predictions, expressed as a percentage of the actual values. It's the mean of the absolute values of the individual percentage errors.
- Interpretation: In real estate valuation, MAPE tells you how big your prediction errors are in relation to the actual property values. For example, a MAPE of 5% means that, on average, your predictions are off by 5% from the actual value. It’s useful for understanding the accuracy of predictions in relative terms, but it can be misleading if dealing with values close to zero.

##### MAE: Mean Absolute Error
- Definition: MAE is the average of the absolute differences between the predicted and actual values. Unlike RMSE, it doesn’t square the differences and, therefore, doesn't give a disproportionately high weight to large errors.
- Interpretation: MAE provides a straightforward measure of average error magnitude in your property valuations. Lower values of MAE indicate better prediction accuracy. It’s particularly useful if you want to understand the typical error magnitude without overly penalizing large but rare errors, as RMSE does.

**How to use them for prediction performance**
- RMSE is useful for highlighting large errors, but can be sensitive to outliers.
- MAPE offers a relative measure of error and is easy to interpret, but can be problematic for values close to zero.
- MAE gives a simple average error measure without overemphasizing large errors.

These metrics together will provide a comprehensive view of the predictor performance. RMSE will highlight if there are any large prediction errors, MAPE will give you a sense of the relative accuracy, and MAE will offer a straightforward average error measure.


#### Important elements in the Pipeline

- GradientBoostingRegressor: This is a machine learning algorithm used for regression tasks. It builds an additive model in a forward stage-wise fashion, allowing for the optimization of arbitrary differentiable loss functions. In each stage, regression trees are fit on the negative gradient of the given loss function, which makes the model adaptively focus on areas where the prediction error is large.
TargetEncoder:

- Target encoding: is a technique for encoding categorical variables by replacing the category with the average target value for that category. It is particularly useful when dealing with high cardinality categorical features, where traditional encoding methods like one-hot encoding could lead to high dimensionality. TargetEncoder helps in capturing valuable information about the categories and can lead to improved model performance.

- ColumnTransformer: This is a utility from Scikit-Learn that allows different columns or column subsets of the input dataset to be transformed separately and the features generated by each transformer will be concatenated to form a single feature space. This is useful for heterogeneous datasets where different types of columns (e.g., categorical, numerical) require different preprocessing and transformation steps. It simplifies the process of applying different transformations to different columns, all within a single transformer object that can be integrated seamlessly into a machine learning pipeline.


## 2. API

API that can receive property information and provide accurate valuation predictions. The FastAPI server `(api/api.py)` is a standalone service that loads the trained model and provides endpoints for making predictions.

Some technical requirements implemented to the API are:
- API also runs in Docker.
- API calls/predictions should generate logs using a logger for future model monitoring.
- API uses the FastAPI framework.
- API has a basic security system (API-Keys).


### Instructions for Using the API

**2.1. Start the API Server:**
- from the root directory of the project, run `python api/api.py` using the command line or terminal. This will start the FastAPI server.

**2.2. Verify the API is Running:**
- After running the command, you should see output in the terminal indicating that the server has started. It typically shows the address where the API is hosted, usually http://0.0.0.0:8000.
- The message "(api/api.py): Loaded pre-trained model successfully." confirms that the model was loaded without issues.

**2.3. Accessing the API:**
- You can now access your API through the provided URL. If you're testing locally, it will be http://localhost:8000.
- FastAPI also generates interactive API documentation that you can access by visiting http://localhost:8000/docs in your web browser.

**2.4. Testing the API:**
- To test the API endpoints, you can use the interactive documentation page, curl commands, or tools like Postman to send requests.
- There are two available API keys "your_api_key", "another_api_key"
- an example to try using `curl` command could be:

```
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'api-key: your_api_key' \
  -H 'Content-Type: application/json' \
  -d '{
  "type": "casa",
  "sector": "vitacura",
  "net_usable_area": 152.0,
  "net_area": 257.0,
  "n_rooms": 3.0,
  "n_bathroom": 3.0,
  "latitude": -33.3794,
  "longitude": -70.5447
}'
```

**2.5. Log checking**
- It is possible to check the log file for information on received requests, predictions, and errors with the command `cat api_logs.log`.

**2.6. Run the Docker Container**
In the terminal or in Docker Playground use the command `docker run -dp 0.0.0.0:8000:3000 djcontadorz/the-api-app` for running the Pipeline docker. 

**Accessing the Application**
Once the container is running, you can access "The Pipeline App" through your browser or API client at http://localhost:8000


---


### Reproducibility & Scalability

To ensure **reproducibility**, meaning that this pipeline can be run by someone else and yield the same results, the following practices were implemented:

- Version Control: Git is used for version control of all code, including data processing scripts, model training code, and API implementation. This ensures that changes are tracked, and the codebase can be reverted or branched as necessary.

- Environment Management: Docker is employed to manage the environment in which the code runs. This guarantees that anyone running the code does so in an identical setup, mitigating issues arising from different software versions or configurations.

- Usage of Full Paths for File Loading: The API has been designed to utilize full paths when loading files. This approach ensures consistent behavior regardless of the directory from which the script is executed, contributing to the reproducibility across different environments.


To ensure **scalability**, i.e., the capability of the pipeline to handle increasing loads (in terms of data volume or number of users) without performance degradation, the following strategies are utilized:

- Microservices Architecture: The system is designed using a microservices architecture, where different components (such as data ingestion, model training, and prediction API) are developed and deployed as separate, loosely coupled services. This architectural choice facilitates easier scaling of individual components in response to varying demand.

- Containerization: Docker is used for containerizing the application. Containerization not only aids in environment management but also simplifies deployment and scaling on cloud platforms. Containers can be easily replicated and managed, making it straightforward to scale the application horizontally to meet increased demand or workload.


### Client database abstraction through Connection Pooling
Database connection pooling is a technique used to manage and reuse database connections efficiently. It's not an abstraction layer between this application and the database; instead, it's a way to optimize database access.
Connection pooling significantly improves performance, especially under high loads, by reducing the overhead of establishing and closing database connections. It's more about performance optimization rather than abstraction.
The ideal use case for this database connection is when the application frequently opens and closes connections to the database. Implementing it minimizes the overhead associated with these operations, thus enhancing the overall efficiency and responsiveness of the application.

### Error handling
Error handling mechanisms are implemented at various stages of the pipeline to ensure robustness and reliability. Key checkpoints include:

- Data Loading: Error handling during data loading ensures that issues like missing files, incorrect formats, or corrupted data are caught early and managed appropriately.
- Model Training: During model training, error handling is crucial for dealing with scenarios like algorithmic convergence issues, data inconsistencies, or unexpected data values.
- API Layer: The API layer includes error handling to manage invalid requests, server errors, and other potential runtime issues that could affect the user experience.

### Project Assumptions

**Data Assumptions**
- The project assumes the availability of training and testing data, either in CSV files or via a provided database connection.
- Data does not contain sensitive or personally identifiable information (PII).
- Handling of missing or NaN values in data is the responsibility of the user.

**Modeling and Pipeline Assumptions**
- A Gradient Boosting Regressor model is used for property valuation prediction.
- Model hyperparameters are based on initial assumptions and can be fine-tuned.
- Relevant features for property valuation include categorical columns such as "type" and "sector."

**API and Logging Assumptions**
- API includes endpoints for receiving property information and returning predictions.
- Basic security measures, such as API keys are enough for this first project iteration.
- the user will provide all necesary features information for the model prediction, so there is no implementation of error handling for the user input data at the moment.

**User Knowledge Assumptions**
- Users are assumed to have basic knowledge of Python and machine learning concepts.



### Suggestions for improvement

This machine learning project addresses the specific needs of our real estate client in Chile by delivering an accurate, efficient, and deployable solution for property valuation. The combination of a robust pipeline and an API enhances the overall usability and accessibility of the model, providing a valuable tool for informed decision-making. Suggestions that can be implemented for continous improvement are:

- **Hyperparameter Tuning:** Perform a hyperparameter tuning process, such as GridSearchCV, to optimize the GradientBoostingRegressor model. Parameters like "learning_rate," "n_estimators," and "max_depth" should be fine-tuned to achieve better model performance.

- **Model Evaluation:** Ensure that the model is not underfitting or overfitting during the evaluation process. This will help in assessing the model's generalization ability and making necessary adjustments.

- **Evaluation Metrics Alignment:** Make sure that the evaluation metrics used (RMSE, MAPE, MAE) align with the client's objectives and business requirements. Customized metrics may be required based on specific goals.

- **Feature Engineering:** Explore additional feature engineering techniques to enhance the model's predictive capabilities. Feature engineering can lead to better insights and improved model performance.

- **GridSearchCV for Model Selection:** Consider using GridSearchCV not only for hyperparameter tuning but also for model selection. Explore different algorithms and configurations to identify the most suitable model for the problem.
# Tarea2_Mlflow_INF3862
