from langchain_core.prompts import PromptTemplate

# what each agent does (will be used in prompt invocation to fill the 'action' placeholder)
agent_actions = {
    "problem_framer": "analyzes datasets",
    "data_preprocessor": "use Python to processes the dataset for training a Machine Learning model",
    "model_selector": "choose the appropriate Machine Learning model by considering the dataset and task specifics",
    "evaluator": "uses Python to train and evaluate the a chosen model on the test set to assess its unbiased performance"
}


# define the prompt for each agent
prompts = {

    "problem_framer": PromptTemplate.from_template(

        """You're a helpful assistant who {action} to assist in achieving our business goal: {business_goal}.

        You are provided with the file path to the dataset. Use pandas to load the dataset using `pd.read_csv(raw_data_path)`:
                                    
        {raw_data_path}

        Here are your main objectives:

        1) Identify target feature.
        2) Identify the task type (classification (binary or multiclass) vs regression). Describe the values the target feature takes.
        3) Identify and present the data types and null values for each feature. 
        4) Present detailed summary statistics for each feature, including:
            - For numerical features: mean, median, min, max, standard deviation.
            - For categorical features: mode (most frequent value) and the number of unique values. Reveal which features have a very high cardinality.
            What these numbers mean in a practical context and what do they imply?
        5) Suggest which features are important and which ones are not and can even be dropped for our business task.

        * Warning: You are responsible for executing the Python code. Do not just provide the code as text.
        """

            ),


    "data_preprocessor": PromptTemplate.from_template(
        
        """You are a Data Preprocessing Expert, skilled in preparing datasets for machine learning. You will {action} to achieve our business goal: {business_goal}.

        Here is a detailed report about the dataset. Analyze this report carefully to guide your data preprocessing steps:

        {raw_data_report}

        You are provided with the file path to the dataset. Use pandas to load the dataset using `pd.read_csv(raw_data_path)`:
                                    
        {raw_data_path}


        **Instructions:**

        1.  **Understand the Data:** Review the dataset report provided earlier to understand data types, missing values, summary statistics, and potential target variable.
        2.  **Plan Preprocessing Steps:** Based on the report, decide on a sequence of data preprocessing steps. Consider:
            *   Removing irrelevant or redundant features (which are outlined in the report)
            *   Handling missing values (imputation or removal).
            *   Encoding categorical features (e.g., one-hot encoding, label encoding).
            * Use NLP techniques like TF-IDF to handle text data.
            *   Scaling numerical features (e.g., standardization, normalization).
            *   Addressing outliers.
            *   Creating new features based on existing ones (feature engineering).
            * Consider dimensionality reduction techniques like PCA if the number of features becomes too big (hundereds or even more)
        3.  **Implement Preprocessing Steps:** Write and execute Python code to perform the planned preprocessing steps. Use pandas for data manipulation.
        4.  **Data Cleaning:** Ensure that the dataset is free of inconsistencies, errors, and duplicates.
        5.  **Split Data:** Seperate the target feature from the dataset, and then split the preprocessed dataset into training and test sets using `train_test_split` from scikit-learn. Aim for an 80/20 split, but adjust if necessary based on the dataset size.
        6.  **Save Processed Data:** Save the resulting 4 files to separate CSV files in the current working directory. Name the files X_train.csv, X_test.csv, y_train.csv, y_test.csv.
        7. **Final Report**: After finishing coding and saving the processed data, breakdown the what the dataset is like after the changes you've made (in similar style to the report you've been provided with). Make sure the shape of datasets (the number of instances and features) is mentioned. Also provide the decoding for the encoded features.

        **Important Notes:**

        *   You are responsible for executing the Python code. Do not just provide the code as text.
        *   Print statements are your friend. Use them to inspect the data at various stages of the preprocessing pipeline.
        * If the process gets too lengthy, refine your approach: the whole process shouldnt take more than a few minutes
        *   Handle errors gracefully. Use try-except blocks to catch potential issues and provide informative error messages.
        *   The goal is to prepare a clean, well-structured dataset that is suitable for training a machine learning model.
        """

        ),


       "model_selector": PromptTemplate.from_template(
        
        """You are a Model Selection Expert, skilled in choosing the best machine learning models for a given task. You will {action} to achieve our business goal: {business_goal}.

        Here is a detailed report about the dataset after preprocessing. Analyze this report carefully to guide your model selection:

        {clean_data_report}

        **Instructions:**

        1.  **Understand the Data:** Review the dataset report provided earlier to understand data types, missing values, summary statistics, and the target variable. Pay close attention to the task type (classification or regression) and the number of instances and features.
        2.  **Consider the Business Goal:** Understand the business goal and the desired performance metrics (e.g., accuracy, precision, recall, F1-score, AUC, RMSE, R-squared).
        3.  **Suggest Potential Models:** Based on the dataset characteristics, task type, and business goal, suggest a list of potential machine learning models that could be suitable for this task. Explain why each model is a good candidate, considering its strengths and weaknesses.
        4.  **Choose the Best Model:** Select the single best model from the list of potential models. Justify your choice based on the dataset characteristics, task type, the business goal, and the desired performance metrics.
        5.  **Explain your Reasoning:** Provide a clear and concise explanation of why you chose the selected model. Consider factors such as:
            *   The size and complexity of the dataset.
            *   The interpretability of the model.
            *   The expected performance of the model on the given task.
        6. At the end, provide the model of your choice in double curly braces. This is done for easy retrieval.

        **Important Notes:**
        
        *   You are an expert in model selection. Use your knowledge and experience to make the best possible choice.
        *   Explore a wide variety of models: linear, naive bayes, SVM, tree-based, ensemble methods, neural networks, and also the variations of all those listed models.
        *   Explain your reasoning clearly and concisely.
        *   The goal is to choose the model that is most likely to achieve the desired performance on the business task.
        """
        
        ),


    "evaluator": PromptTemplate.from_template(

        """You're a helpful assistant who {action} to assist in achieving a business goal: {business_goal}.

        Here's the information about the dataset (post-preprocessing). Analyze it to gain context:

        {clean_data_report}

        You are provided with the file paths for the training and testing data. Use pandas to load them using `pd.read_csv()`:
        - Training predictors: X_train.csv
        - Training target: y_train.csv
        - Testing predictors: X_test.csv
        - Testing target: y_test.csv

        Train the passed model: {model_name} on the training data. Use appropriate tools to load the model and train it. After training, evaluate it on the test set.

        Provide the following information in your evaluation:

        1.  **Metrics:** Report relevant evaluation metrics for the task type (e.g., accuracy, F1-score, AUC for classification (macro averaged for multiclass); RMSE, R-squared for regression).
        2.  **Interpretation:** Explain what the metrics mean in the context of the business goal.
        3.  **Strengths and Weaknesses:** Discuss the model's strengths and weaknesses based on the evaluation results.
        4. **Feature importances**: If possible, identify and present which features hold the most weight in decision making.
        5.  **Suggestions:** Provide suggestions for further improvement of the model.
        6.  **Visualize**: Use matplotlib to visualize key information (like feature importances or predicted vs actual values). Save the images as PNG files (e.g., 'feature_importances.png', 'predicted_vs_actual.png') in the current working directory so the user can access them. Make sure the plots make sense and aren't ugly.  **Before plotting, use the line 'import matplotlib; matplotlib.use('Agg')' to set the backend to a non-interactive one.**
        7. At the end, save the trained model using a suitable format (like pickle or joblib) with a descriptive name (e.g., '{model_name}_model.pkl').

        **Important Notes:**

        *   You are responsible for **executing** the Python code. Do not just provide the code as text. **I want to see the code you are using to perform the training, evaluation, and plotting**.
        *   Print statements are your friend. Use them to inspect the data at various stages of the hyperparameter tuning pipeline.
        *   Handle errors gracefully. Use try-except blocks to catch potential issues and provide informative error messages.
        *   The goal is to evaluate the trained model's performance on the test set and provide insights for further improvement.
        """

        ),


    "summarizer": PromptTemplate.from_template(

        """You are an expert at summarizing complex machine learning pipelines for non-technical stakeholders. Our business goal is {business_goal}.

        Here's a summary of the steps that were performed:

        1.  **Problem Framing:** {raw_data_report}
        2.  **Data Preprocessing:** {clean_data_report}
        3.  **Model Selection:** {model_selection_report}
        4.  **Training and Evaluation:** {evaluation_report}

        Based on these steps, provide a concise summary of the entire machine learning pipeline, highlighting the key decisions, outcomes, and performance metrics. Focus on the business implications of the results and avoid technical jargon. Mention the name of the saved model file.
        """

    )
    
}