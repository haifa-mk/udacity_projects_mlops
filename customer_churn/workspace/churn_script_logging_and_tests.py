
"""
churn_script_logging_and_tests.py
Author: Haifa 
Date: 2025-11-08

This script performs testing and logging for all functions in churn_library.py.
Each test verifies that the corresponding function executes successfully and 
saves expected outputs (dataframes, plots, models). All logs are stored in 
'./logs/churn_library.log'.
"""

import os
import logging
import pandas as pd
import pytest
from churn_library import import_data, perform_eda, perform_feature_engineering, \
    encoder_helper, train_models

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="session")
def df():
    """Load the dataset once for all tests."""
    try:
        df = import_data("./data/bank_data.csv")
        assert df.shape[0] > 0
        logging.info(
            "Fixture data_df: Loaded successfully with shape %s", df.shape)
        return df
    except Exception as err:
        logging.error("Fixture data_df: Failed to load dataset: %s", err)
        raise


def test_import(df):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        expected_files = [
            "churn_distribution.png",
            "age_distribution.png",
            "marital_status_distribution.png",
            "total_transaction_distribution.png",
            "heatmap.png"
        ]
        for file in expected_files:
            path = os.path.join("images/eda", file)
            assert os.path.exists(path)

        logging.info("Testing perform_eda: All plots saved successfully.")
    except Exception as err:
        logging.error("Testing perform_eda: Failed to save EDA plots")
        raise err


def test_encoder_helper(df):
    '''
    test encoder helper
    '''
    cat_cols = df.select_dtypes(include=['object']).columns
    df["Churn"] = df["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    try:
        df_encoded = encoder_helper(df, cat_cols, response="Churn")
        # Check if new encoded columns exist
        for col in cat_cols:
            assert f"{col}_Churn" in df_encoded.columns
        logging.info(
            "Testing encoder_helper: Encoded columns added successfully.")
    except Exception as err:
        logging.error("Testing encoder_helper: Encoding failed ")
        raise err


def test_perform_feature_engineering(df):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, y_train, X_test, y_test, X_df = perform_feature_engineering(
            df, "Churn")
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert isinstance(X_df, pd.DataFrame)
        logging.info(
            "Testing perform_feature_engineering: Data split successful.")
    except Exception as err:
        logging.error("Testing perform_feature_engineering: Failed")
        raise err


def test_train_models(df):
    '''
    test train_models
    '''
    try:
        X_train, y_train, X_test, y_test, X_df = perform_feature_engineering(
            df, "Churn")
        train_models(X_df, X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: Models trained and saved.")
    except Exception as err:
        logging.exception("Testing train_models: Model training failed")
        raise err

    #  Verify saved models
    try:
        assert os.path.exists("./models/rfc_model.pkl")
        logging.info("train_models: RandomForest model saved successfully.")
    except AssertionError:
        logging.exception("train_models: RandomForest model not found.")

    try:
        assert os.path.exists("./models/logistic_model.pkl")
        logging.info(
            "train_models: Logistic Regression model saved successfully.")
    except AssertionError:
        logging.exception("train_models: Logistic Regression model not found.")
        raise

    result_images = [
        "images/results/logistic_results.png",
        "images/results/rf_results.png",
        "images/results/feature_importance.png",
        "images/results/roc_curve.png"
    ]
    for img in result_images:
        try:
            assert os.path.exists(img)
            logging.info("train_models: Output image exists — %s", img)
        except AssertionError:
            logging.exception("train_models: Missing result image — %s", img)
            raise


if __name__ == "__main__":
    pytest.main(["-v", "churn_script_logging_and_tests.py"])
