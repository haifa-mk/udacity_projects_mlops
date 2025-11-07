# library doc string
"""
churn_library.py
Author: Haifa
Date: 2025-11-08
Utilities for importing data, performing exploratory data analysis (EDA),
feature engineering, model training, evaluation, and visualization for a
customer churn prediction pipeline.

"""


# import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, RocCurveDisplay, roc_auc_score
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def create_and_save_plot(
    plot_func,
    file_name,
    save_dir="images/eda",
    title=None,
    xlabel=None,
    ylabel=None,
    figsize=(
        20,
        10)):
    '''
    Helper function to handle figure creation, styling, labeling, and saving.

    input:
            plot_func : function that performs the actual plotting (e.g., lambda: plt.bar(...))
            file_name : output filename (string)
            save_dir  : directroy to save image in
            title     : optional plot title (string)
            xlabel    : optional label for x-axis (string)
            ylabel    : optional label for y-axis (string)
            figsize   : tuple defining figure size (default: (20, 10))
    output:
            None
    '''
    plt.figure(figsize=figsize)
    plt.style.use('seaborn-v0_8-darkgrid')
    plot_func()

    if title:
        plt.title(title)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name),
                dpi=300, bbox_inches="tight")
    plt.close()


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.style.use('seaborn-v0_8-darkgrid')
    # Plot 1: Churn distribution
    # Encode 'Attrition_Flag' as 0 for existing and 1 for churned customers
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    create_and_save_plot(lambda: df["Churn"].hist(
    ), file_name="churn_distribution.png", title="Customer Churn Distribution")
    # Plot 2: Age disribution
    create_and_save_plot(lambda: df["Customer_Age"].hist(
    ), file_name='age_distribution.png', title="Age Distribution")
    # Plot 3: Marital status disribution
    create_and_save_plot(
        lambda: df.Marital_Status.value_counts('normalize').plot(
            kind='bar'),
        file_name='marital_status_distribution.png',
        title="Marital Status Distribution")
    # Plot 4: Marital status disribution
    create_and_save_plot(
        lambda: sns.histplot(
            df['Total_Trans_Ct'],
            stat='density',
            kde=True),
        file_name="total_transaction_distribution.png",
        title="Total Transaction Distribution")
    # Plot 5 : heat map
    create_and_save_plot(
        lambda: sns.heatmap(
            df.select_dtypes(
                include=['number']).corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2),
        file_name="heatmap.png")


def encoder_helper(df, category_lst, response="Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for category in category_lst:
        cat_group = df.groupby(category)["Churn"].mean()
        df[f"{category}_{response}"] = df[category].map(cat_group)

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name
              [optional argument that could be used for naming variables or index y column]

    output:
              X_train_scaled: scaled X training data
              X_test_scaled: scaled X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = df.select_dtypes(include=['object', 'category']).columns

    df = encoder_helper(df, cat_columns, response)
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = df[keep_cols]
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    # --- Scale features ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, y_train, X_test_scaled, y_test, X


def classification_report_image(
    model_name,
    y_train, y_train_preds,
    y_test, y_test_preds,
    save_path,
    figsize=(6, 6),
    fontsize=10
):
    '''
   Generates a classification report for training and testing results of a given model,
   displaying metrics such as precision, recall, and F1-score, and saves the report as an image.

   input:
           model_name: string representing the model name (e.g., 'Random Forest')
           y_train: true labels for the training dataset
           y_train_preds: predicted labels for the training dataset
           y_test: true labels for the test dataset
           y_test_preds: predicted labels for the test dataset
           save_path: file path where the classification report image will be saved
           fontsize: integer defining text font size (default: 10)

   output:
           None
   '''

    plt.rc('figure', figsize=figsize)

    # Training results
    plt.text(0.01, 1.25, f'{model_name} Train', {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds)),
             {'fontsize': fontsize}, fontproperties='monospace')

    # Testing results
    plt.text(0.01, 0.6, f'{model_name} Test', {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds)),
             {'fontsize': fontsize}, fontproperties='monospace')

    plt.axis('off')
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    create_and_save_plot(
        lambda: (
            plt.bar(range(X_data.shape[1]), importances[indices]),
            plt.xticks(range(X_data.shape[1]), names, rotation=90)
        ),
        file_name=output_pth,
        title="Feature Importance",
        xlabel="Features",
        ylabel="Importance",
        figsize=(20, 5),
        save_dir="images/results"
    )


def plot_roc_curves(models, X_test, y_test,
                    save_path="images/results/roc_curve.png"):
    '''
    Plots ROC curves for given models and saves the figure.

    input:
        models    : dictionary of models { "name": model_object }
        X_test    : X test data
        y_test    : y test data
        save_path : path to save ROC plot (default: "images/results/roc_curve.png")

    output:
        None
    '''
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    auc_scores = {}

    for name, model in models.items():
        RocCurveDisplay.from_estimator(
            model, X_test, y_test, ax=ax, alpha=0.8, name=name)
        if hasattr(model, "predict_proba"):
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            auc = roc_auc_score(y_test, model.decision_function(X_test))
        auc_scores[name] = auc
        print(f"{name} AUC: {auc:.2f}")

    plt.title("ROC Curve Comparison")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to: {save_path}")


def train_models(X_df, X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_df   : data frame
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': [None, 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(
        estimator=rfc, param_grid=param_grid, cv=5, verbose=2)
    print(f"X_train shape {X_train.shape}")
    print(f"y_train shape {y_train.shape}")
    print("\nTraining Random Forest with GridSearchCV...")

    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')
    classification_report_image(
        model_name="Logistic Regression",
        y_train=y_train,
        y_test=y_test,
        y_train_preds=y_train_preds_lr,
        y_test_preds=y_test_preds_lr,
        save_path="images/results/logistic_results.png")
    classification_report_image(
        model_name="Random Forest",
        y_train=y_train,
        y_test=y_test,
        y_train_preds=y_train_preds_rf,
        y_test_preds=y_test_preds_rf,
        save_path="images/results/rf_results.png")

    feature_importance_plot(cv_rfc, X_df, "feature_importance.png")
    plot_roc_curves(
        {"Random Forest": cv_rfc.best_estimator_, "Logistic Regression": lrc},
        X_test,
        y_test,
        save_path="images/results/roc_curve.png"
    )


if __name__ == "__main__":

    df = import_data("data/bank_data.csv")
    print(df.head(3))
    perform_eda(df)
    X_train, y_train, X_test, y_test, X_df = perform_feature_engineering(
        df, "Churn")
    train_models(X_df, X_train, X_test, y_train, y_test)
