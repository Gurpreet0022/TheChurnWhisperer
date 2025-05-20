import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
import xgboost as xgb
import pickle
import os
from warnings import filterwarnings
filterwarnings('ignore')


# Set page configuration
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add an anchor at the top of the page
st.markdown('<a id="top"></a>', unsafe_allow_html=True)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E3A8A;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #3B82F6;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .feature-importance {
        height: 400px;
    }
</style>
""", unsafe_allow_html=True)

# Define functions for each part of the application
def load_data():
    """Load the telecom customer churn dataset."""
    # Check if the file exists in the current directory
    if os.path.exists('Telco_Churn_Clean_data.csv'):
        df = pd.read_csv('Telco_Churn_Clean_data.csv')
    else:
        # For demo purposes, create a synthetic dataset based on typical telecom churn data
        st.warning("No dataset found. Using synthetic data for demonstration.")
        
        df_features = df_features.convert_dtypes()          

        # Generate synthetic data
        np.random.seed(42)
        n_samples = 5000
        
        # Generate customer IDs
        customer_ids = [f'CUST{i:05d}' for i in range(1, n_samples + 1)]
        
        # Generate features
        genders = np.random.choice(['Male', 'Female'], size=n_samples)
        senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
        partner = np.random.choice(['Yes', 'No'], size=n_samples)
        dependents = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
        tenure = np.random.gamma(2, 18, n_samples).astype(int)
        tenure = np.clip(tenure, 0, 72)  # Clip tenure to 0-72 months
        
        phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
        multiple_lines = []
        for ps in phone_service:
            if ps == 'No':
                multiple_lines.append('No phone service')
            else:
                multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
        
        internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.4, 0.4, 0.2])
        
        # Generate additional services based on internet service
        online_security = []
        online_backup = []
        device_protection = []
        tech_support = []
        streaming_tv = []
        streaming_movies = []
        
        for i_service in internet_service:
            if i_service == 'No':
                online_security.append('No internet service')
                online_backup.append('No internet service')
                device_protection.append('No internet service')
                tech_support.append('No internet service')
                streaming_tv.append('No internet service')
                streaming_movies.append('No internet service')
            else:
                online_security.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
                online_backup.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
                device_protection.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
                tech_support.append(np.random.choice(['Yes', 'No'], p=[0.3, 0.7]))
                streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.5, 0.5]))
                streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.5, 0.5]))
        
        contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                  size=n_samples, p=[0.6, 0.2, 0.2])
        paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples)
        payment_method = np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            size=n_samples, p=[0.4, 0.15, 0.25, 0.2]
        )
        
        # Generate charges based on services
        monthly_charges = []
        for i in range(n_samples):
            base_charge = 20
            if phone_service[i] == 'Yes':
                base_charge += 25
                if multiple_lines[i] == 'Yes':
                    base_charge += 15
            
            if internet_service[i] == 'DSL':
                base_charge += 35
            elif internet_service[i] == 'Fiber optic':
                base_charge += 65
                
            for service in [online_security[i], online_backup[i], device_protection[i], 
                           tech_support[i], streaming_tv[i], streaming_movies[i]]:
                if service == 'Yes':
                    base_charge += 8
                    
            # Add some random noise
            base_charge += np.random.normal(0, 5)
            monthly_charges.append(max(0, base_charge))
        
        # Total charges as a function of tenure and monthly charges
        total_charges = []
        for i in range(n_samples):
            # Add some variability to represent changes in monthly charges over time
            total_charges.append(tenure[i] * monthly_charges[i] * np.random.uniform(0.9, 1.1))
            
        # Generate churn based on features
        churn_prob = []
        for i in range(n_samples):
            # Base probability
            prob = 0.15
            
            # Contract effect
            if contract[i] == 'Month-to-month':
                prob += 0.15
            elif contract[i] == 'One year':
                prob -= 0.05
            elif contract[i] == 'Two year':
                prob -= 0.1
                
            # Tenure effect (longer tenure = lower churn)
            prob -= min(0.15, tenure[i] / 200)
            
            # Service quality factors
            if tech_support[i] == 'No' and internet_service[i] != 'No':
                prob += 0.05
            if online_security[i] == 'No' and internet_service[i] != 'No':
                prob += 0.03
                
            # Payment method effect
            if payment_method[i] == 'Electronic check':
                prob += 0.05
                
            # Price sensitivity
            if monthly_charges[i] > 80:
                prob += 0.05
                
            # Senior citizen effect
            if senior_citizen[i] == 1:
                prob += 0.03
                
            # Add randomness
            prob = min(0.9, max(0.05, prob + np.random.normal(0, 0.1)))
            churn_prob.append(prob)
            
        churn = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in churn_prob]
            
        # Create DataFrame
        df = pd.DataFrame({
            'customerID': customer_ids,
            'gender': genders,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Churn': churn
        })
        
        # Convert TotalCharges to string to simulate loading issues that might occur with real data
        df['TotalCharges'] = df['TotalCharges'].astype(str)
        
        # Save the synthetic dataset for future use
        df.to_csv('telecom_churn_data.csv', index=False)
    
    return df

def preprocess_data(df):
    """Preprocess the data for analysis and modeling."""
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert TotalCharges to numeric, handling potential errors
    df_processed['TotalCharges'] = pd.to_numeric(df_processed['TotalCharges'], errors='coerce')
    
    # Handle any missing values in TotalCharges
    df_processed['TotalCharges'] = df_processed['TotalCharges'].fillna(df_processed['MonthlyCharges'])

    
    # Convert SeniorCitizen from 0/1 to No/Yes for consistency
    df_processed['SeniorCitizen'] = df_processed['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    return df_processed

def create_features(df):
    """Create additional features for improved model performance."""
    df_features = df.copy()
    
    # Create a tenure group feature
    df_features['tenure_group'] = pd.cut(
        df_features['tenure'],
        bins=[0, 12, 24, 36, 48, 60, 72],
        labels=['0-12 mo', '13-24 mo', '25-36 mo', '37-48 mo', '49-60 mo', '61-72 mo']
    )
    
    # Calculate Monthly to Total Charges Ratio (customer consistency)
    df_features['charge_ratio'] = df_features['MonthlyCharges'] * df_features['tenure'] / df_features['TotalCharges']
    df_features['charge_ratio'] = df_features['charge_ratio'].fillna(1) # Handle division by zero

    df_features['charge_ratio'] = df_features['charge_ratio'].clip(0.5, 1.5)  # Clip outliers
    
    # Count the number of additional services
    service_columns = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df_features['additional_services'] = df_features[service_columns].apply(
        lambda row: sum(1 for item in row if item == 'Yes'), axis=1
    )
    
    # Create a binary flag for any streaming service
    df_features['has_streaming'] = ((df_features['StreamingTV'] == 'Yes') | 
                                   (df_features['StreamingMovies'] == 'Yes')).astype(int)
    
    # Create a binary flag for any protection service
    df_features['has_protection'] = ((df_features['OnlineSecurity'] == 'Yes') | 
                                    (df_features['OnlineBackup'] == 'Yes') |
                                    (df_features['DeviceProtection'] == 'Yes') |
                                    (df_features['TechSupport'] == 'Yes')).astype(int)
    
    # Create a binary flag for automatic payment methods
    df_features['auto_payment'] = df_features['PaymentMethod'].apply(
        lambda x: 1 if 'automatic' in x else 0
    )
    
    # Average monthly charge per service
    df_features['avg_service_cost'] = df_features['MonthlyCharges'] / (df_features['additional_services'] + 1)
    
    return df_features

def prepare_modeling_data(df):
    """Prepare the data for modeling by separating features and target."""
    # Drop unnecessary columns
    df_model = df.copy()
    df_model = df_model.drop(['customerID', 'tenure_group'], axis=1, errors='ignore')
    
    # Convert target to binary
    df_model['Churn'] = df_model['Churn'].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return X, y, categorical_cols, numerical_cols

def create_model_pipeline(categorical_cols, numerical_cols):
    """Create a preprocessing and modeling pipeline."""
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Create the modeling pipeline with the selected classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    return model_pipeline

def train_and_evaluate_model(X, y, model_pipeline, model_type='random_forest'):
    """Train and evaluate the machine learning model."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define the model based on user selection
    if model_type == 'random_forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    elif model_type == 'xgboost':
        model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
    
    # Update the classifier in the pipeline
    model_pipeline.steps[-1] = ('classifier', model)
    
    # Create a simplified grid search for demonstration purposes
    grid_search = GridSearchCV(
        model_pipeline, 
        param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    # Train the model
    with st.spinner('Training the model... This may take a few minutes.'):
        grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Get predictions on the test set
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    cls_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Save the model
    with open(f'churn_prediction_{model_type}.pkl', 'wb') as file:
        pickle.dump(best_model, file)
    
    return best_model, X_test, y_test, y_pred, y_prob, cls_report, conf_matrix, grid_search.best_params_

def display_model_performance(model, X_test, y_test, y_pred, y_prob, cls_report, conf_matrix, feature_names, model_type):
    """Display the model performance metrics and visualizations."""
    col1, col2 = st.columns(2)
        
    with col1:
        st.markdown("<div class='sub-header'>Classification Report</div>", unsafe_allow_html=True)
        
        # Format the classification report
        report_df = pd.DataFrame(cls_report).transpose()
        if '0' in report_df.columns and '1' in report_df.columns:
            report_df = report_df.drop(['0', '1'], axis=1, errors='ignore')
        
        # Rename the index for clarity
        if 0 in report_df.index and 1 in report_df.index:
            report_df = report_df.rename(index={0: 'Not Churned', 1: 'Churned'})
        
        # Display the report
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
        
        st.markdown("<div class='sub-header'>Confusion Matrix</div>", unsafe_allow_html=True)
        
        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Not Churned', 'Churned'], 
                   yticklabels=['Not Churned', 'Churned'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
    
    with col2:
        st.markdown("<div class='sub-header'>ROC Curve</div>", unsafe_allow_html=True)
        
        # Plot the ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        st.markdown("<div class='sub-header'>Precision-Recall Curve</div>", unsafe_allow_html=True)
        
        # Plot the precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        st.pyplot(fig)
    
    # Display feature importance
    st.markdown("<div class='sub-header'>Feature Importance</div>", unsafe_allow_html=True)
    
    # Extract the classifier from the pipeline
    classifier = model.named_steps['classifier']
    
    # Get the preprocessor
    preprocessor = model.named_steps['preprocessor']
    
    # Transform the feature names
    feature_names_out = []
    
    # Get output feature names from the preprocessor if available
    if hasattr(preprocessor, 'get_feature_names_out'):
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except:
            # For older scikit-learn versions or if get_feature_names_out fails
            st.warning("Unable to extract exact feature names from the preprocessor.")
            feature_names_out = [f"feature_{i}" for i in range(100)]  # Use generic names
    else:
        st.warning("Preprocessor doesn't support get_feature_names_out")
        feature_names_out = [f"feature_{i}" for i in range(100)]
    
    # Plot feature importance based on model type
    if model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
        # Extract feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            if len(importances) == len(feature_names_out):
                # Create a DataFrame for the feature importances
                importance_df = pd.DataFrame({
                    'feature': feature_names_out,
                    'importance': importances
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False).head(15)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))       #10,8
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
                ax.set_title('Top 15 Feature Importances')
                st.pyplot(fig)
            else:
                st.warning(f"Feature importance shape mismatch: {len(importances)} importances vs {len(feature_names_out)} feature names")
        else:
            st.warning("This model doesn't provide feature importances directly.")
    
    elif model_type == 'logistic_regression':
        if hasattr(classifier, 'coef_'):
            # For logistic regression, we use the absolute values of coefficients
            coefs = classifier.coef_[0]
            if len(coefs) == len(feature_names_out):
                # Create a DataFrame for the coefficients
                coef_df = pd.DataFrame({
                    'feature': feature_names_out,
                    'coefficient': np.abs(coefs)
                })
                
                # Sort by absolute coefficient value
                coef_df = coef_df.sort_values('coefficient', ascending=False).head(15)
                
                # Plot
                fig, ax = plt.subplots(figsize=(10, 8))        #10,8
                sns.barplot(x='coefficient', y='feature', data=coef_df, ax=ax)
                ax.set_title('Top 15 Feature Coefficients (Absolute Value)')
                st.pyplot(fig)
            else:
                st.warning(f"Coefficient shape mismatch: {len(coefs)} coefficients vs {len(feature_names_out)} feature names")
        else:
            st.warning("Unable to extract coefficients from the model.")

def customer_churn_prediction(model, X):
    """Make predictions using the trained model for a new customer."""
    # Make prediction
    prediction_proba = model.predict_proba(X.iloc[[0]])[:, 1][0]
    prediction = model.predict(X.iloc[[0]])[0]
    
    return prediction, prediction_proba

def get_churn_recommendations(df, features, prediction_proba):
    """Generate customized recommendations based on customer profile and churn risk."""
    recommendations = []
    
    # Define risk categories
    risk_level = "Low"
    if prediction_proba > 0.3 and prediction_proba <= 0.6:
        risk_level = "Medium"
    elif prediction_proba > 0.6:
        risk_level = "High"
    
    # Contract-based recommendations
    if features['Contract'].iloc[0] == 'Month-to-month':
        recommendations.append("Offer a discount for upgrading to a one-year or two-year contract")
    
    # Service-based recommendations
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    missing_services = [service for service in services if features[service].iloc[0] == 'No']
    
    if len(missing_services) > 0:
        service_names = {
            'OnlineSecurity': 'Online Security', 
            'OnlineBackup': 'Online Backup',
            'DeviceProtection': 'Device Protection',
            'TechSupport': 'Tech Support'
        }
        
        # Find similar customers who have these services and don't churn
        if 'TechSupport' in missing_services and features['InternetService'].iloc[0] != 'No':
            recommendations.append("Offer free Tech Support trial - customers with this service are 36% less likely to churn")
        
        if 'OnlineSecurity' in missing_services and features['InternetService'].iloc[0] != 'No':
            recommendations.append("Highlight Online Security benefits - customers with this service are 45% less likely to churn")
    
    # Payment method recommendations
    if features['PaymentMethod'].iloc[0] == 'Electronic check':
        recommendations.append("Offer incentives to switch to automatic payment methods, which are associated with higher retention")
    
    # Pricing recommendations based on tenure and charges
    if features['MonthlyCharges'].iloc[0] > 80 and features['tenure'].iloc[0] < 12:
        recommendations.append("Consider offering a temporary discount or loyalty program for new high-value customers")
    
    # Bundle recommendations
    if features['InternetService'].iloc[0] != 'No' and features['PhoneService'].iloc[0] == 'Yes':
        if features['has_streaming'].iloc[0] == 0:
            recommendations.append("Offer a streaming bundle discount to increase service stickiness")
    
    # Family plan recommendations
    if features['Partner'].iloc[0] == 'Yes' or features['Dependents'].iloc[0] == 'Yes':
        recommendations.append("Offer family plan benefits or multi-line discounts")
    
    # If few recommendations, add general ones
    if len(recommendations) < 2:
        recommendations.append("Engage with personalized communication to understand specific needs")
        if risk_level != "Low":
            recommendations.append("Conduct a satisfaction survey to identify improvement areas")
    
    return risk_level, recommendations

def visualize_customer_insights(df, features):
    """Create visualizations to provide insights about the customer within the context of the dataset."""
    # Tenure comparison
    fig, ax = plt.subplots(figsize=(10,6))         #10,6
    
    # Plot tenure distribution with customer's tenure highlighted
    sns.histplot(df['tenure'], kde=True, ax=ax)
    customer_tenure = features['tenure'].iloc[0]
    ax.axvline(customer_tenure, color='red', linestyle='--', label=f'Customer Tenure: {customer_tenure} months')
    ax.set_title('Tenure Distribution with Customer Position')
    ax.set_xlabel('Tenure (months)')
    ax.legend()
    
    st.pyplot(fig)
    
    # Monthly charges comparison
    fig, ax = plt.subplots(figsize=(10, 6))  #10,6
    
    # Plot charges distribution with customer's charges highlighted
    sns.histplot(df['MonthlyCharges'], kde=True, ax=ax)
    customer_charges = features['MonthlyCharges'].iloc[0]
    ax.axvline(customer_charges, color='red', linestyle='--', label=f'Customer Charges: ${customer_charges:.2f}')
    ax.set_title('Monthly Charges Distribution with Customer Position')
    ax.set_xlabel('Monthly Charges ($)')
    ax.legend()
    
    st.pyplot(fig)
    
    # Services comparison
    services = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Create a comparison table
    comparison_data = []
    
    for service in services:
        if service in features.columns:
            customer_value = features[service].iloc[0]
            
            # Calculate percentage of customers with the same service choice
            total_customers = len(df)
            matching_customers = len(df[df[service] == customer_value])
            percentage = (matching_customers / total_customers) * 100
            
            # Calculate churn rate for this service choice
            churned_customers = len(df[(df[service] == customer_value) & (df['Churn'] == 'Yes')])
            churn_rate = (churned_customers / matching_customers) * 100 if matching_customers > 0 else 0
            
            comparison_data.append({
                'Service': service,
                'Customer Value': customer_value,
                'Customers with Same Choice (%)': f"{percentage:.1f}%",
                'Churn Rate for this Choice (%)': f"{churn_rate:.1f}%"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)


def main():
    st.markdown("<div class='main-header'>Telecom Customer Churn Prediction & Analysis</div>", unsafe_allow_html=True)
    
    # Create tabs for different application sections
    tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Churn Analysis", "Predictive Modeling", "Customer Prediction"])
    
    # Load and preprocess data
    df = load_data()
    df_processed = preprocess_data(df)
    df_features = create_features(df_processed)
    
    # Data Overview Tab
    with tab1:
        st.markdown("<div class='sub-header'>Dataset Overview</div>", unsafe_allow_html=True)
        
        # Display basic statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            churn_rate = df['Churn'].value_counts(normalize=True).get('Yes', 0) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col3:
            avg_tenure = df['tenure'].mean()
            st.metric("Avg. Tenure", f"{avg_tenure:.1f} months")
        with col4:
            avg_monthly = df['MonthlyCharges'].mean()
            st.metric("Avg. Monthly Charges", f"${avg_monthly:.2f}")
        
        # Display sample data
        st.markdown("#### Sample Data")
        st.dataframe(df.head(), use_container_width=True)
        
        # Data quality checks
        st.markdown("#### Data Quality")
        
        col1, col2 = st.columns(2)
        with col1:
            # Missing values
            missing_values = df.isnull().sum()
            if missing_values.sum() > 0:
                st.write("Missing Values:", missing_values[missing_values > 0])
            else:
                st.write("✅ No missing values found!")
                
        with col2:
            # Data types
            st.write("Data Types:")
            st.write(df.dtypes)
        
        # Data distribution
        st.markdown("#### Data Distribution")
        
        # Create distribution plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 10))    
        
        # Tenure distribution
        sns.histplot(df['tenure'], kde=True, ax=axes[0])
        axes[0].set_title('Tenure Distribution')
        
        # Monthly charges distribution
        sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1])
        axes[1].set_title('Monthly Charges Distribution')
        
        # Total charges distribution
        sns.histplot(pd.to_numeric(df['TotalCharges'], errors='coerce'), kde=True, ax=axes[2])
        axes[2].set_title('Total Charges Distribution')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Categorical distributions
        st.markdown("#### Categorical Features Distribution")
        
        categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'InternetService', 'Contract', 'PaymentMethod']
        
        # Create a 2x4 grid for the categorical plots
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(categorical_cols):
            if i < len(axes):
                # Count plot with percentage labels
                counts = df[col].value_counts()
                sns.countplot(x=df[col], ax=axes[i])
                axes[i].set_title(f'{col} Distribution')
                
                # Add percentage labels
                total = len(df[col])
                for p in axes[i].patches:
                    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    axes[i].annotate(percentage, (x, y), ha='center')
                
                if i == len(categorical_cols) - 1:
                    for j in range(i + 1, len(axes)):
                        axes[j].set_visible(False)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    # Churn Analysis Tab
    with tab2:
        st.markdown("<div class='sub-header'>Churn Analysis</div>", unsafe_allow_html=True)
        
        # Churn rate by feature
        st.markdown("#### Churn Rate by Customer Characteristics")
        
        # Select feature for analysis
        feature_options = ['Contract', 'InternetService', 'PaymentMethod', 'tenure_group', 
                          'TechSupport', 'OnlineSecurity', 'PaperlessBilling', 'Partner']
        selected_feature = st.selectbox("Select Feature", options=feature_options)
        
        # Calculate churn rate by the selected feature
        churn_by_feature = df_features.groupby(selected_feature)['Churn'].value_counts(normalize=True).unstack()
        if 'Yes' in churn_by_feature.columns:
            churn_rates = churn_by_feature['Yes'] * 100
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            churn_rates.plot(kind='bar', ax=ax)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title(f'Churn Rate by {selected_feature}')
            
            # Add percentage labels
            for i, v in enumerate(churn_rates):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
                
            st.pyplot(fig)
            
            # Display the data in a table
            st.dataframe(pd.DataFrame({
                selected_feature: churn_rates.index,
                'Churn Rate (%)': churn_rates.values.round(1),
                'Customer Count': df_features[selected_feature].value_counts().reindex(churn_rates.index).values
            }), use_container_width=True)
        
        # Monthly charges vs. churn
        st.markdown("#### Monthly Charges vs. Churn")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df_features, ax=ax)
        ax.set_title('Monthly Charges Distribution by Churn Status')
        st.pyplot(fig)
        
        # Churn rate by tenure
        st.markdown("#### Churn Rate by Tenure")
        
        tenure_churn = df_features.groupby('tenure_group', observed=False)['Churn'].value_counts(normalize=True).unstack()

        if 'Yes' in tenure_churn.columns:
            tenure_churn_rate = tenure_churn['Yes'] * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            tenure_churn_rate.plot(kind='bar', ax=ax)
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Tenure Group')
            
            # Add percentage labels
            for i, v in enumerate(tenure_churn_rate):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
                
            st.pyplot(fig)
        
        # Service adoption and churn
        st.markdown("#### Service Adoption and Churn")
        
        # Calculate churn rate by number of additional services
        services_churn = df_features.groupby('additional_services')['Churn'].value_counts(normalize=True).unstack()
        if 'Yes' in services_churn.columns:
            services_churn_rate = services_churn['Yes'] * 100
            
            fig, ax = plt.subplots(figsize=(10, 6))
            services_churn_rate.plot(kind='bar', ax=ax)
            ax.set_xlabel('Number of Additional Services')
            ax.set_ylabel('Churn Rate (%)')
            ax.set_title('Churn Rate by Number of Additional Services')
            
            # Add percentage labels
            for i, v in enumerate(services_churn_rate):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
                
            st.pyplot(fig)
            
        # Protection services and churn
        st.markdown("#### Protection Services and Churn")
        
        services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        service_churn_rates = []
        
        for service in services:
            # Calculate churn rate by service
            service_data = df_features[df_features['InternetService'] != 'No']  # Only include customers with internet
            churn_by_service = service_data.groupby(service)['Churn'].value_counts(normalize=True).unstack()
            
            if 'Yes' in churn_by_service.columns and 'No' in churn_by_service.index:
                # Extract churn rate for customers without the service
                churn_rate_no_service = churn_by_service.loc['No', 'Yes'] * 100
                service_churn_rates.append({
                    'Service': service,
                    'Churn Rate Without Service (%)': churn_rate_no_service
                })
        
        if service_churn_rates:
            service_df = pd.DataFrame(service_churn_rates)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Service', y='Churn Rate Without Service (%)', data=service_df, ax=ax)
            ax.set_title('Churn Rate for Customers Without Protection Services')
            ax.set_ylim(0, 100)
            
            # Add percentage labels
            for i, v in enumerate(service_df['Churn Rate Without Service (%)']):
                ax.text(i, v + 1, f'{v:.1f}%', ha='center')
                
            st.pyplot(fig)
    
    # Predictive Modeling Tab
    with tab3:
        st.markdown("<div class='sub-header'>Predictive Modeling</div>", unsafe_allow_html=True)
        
        # Prepare data for modeling
        X, y, categorical_cols, numerical_cols = prepare_modeling_data(df_features)
        
        # Model selection
        st.markdown("#### Model Selection")
        model_type = st.selectbox(
            "Select Model Type", 
            options=['random_forest', 'gradient_boosting', 'xgboost', 'logistic_regression'],
            format_func=lambda x: {
                'random_forest': 'Random Forest',
                'gradient_boosting': 'Gradient Boosting',
                'xgboost': 'XGBoost',
                'logistic_regression': 'Logistic Regression'
            }[x]
        )
        
        # Create model pipeline
        model_pipeline = create_model_pipeline(categorical_cols, numerical_cols)
        
        # Train button
        if st.button('Train Model'):
            # Train and evaluate the model
            model, X_test, y_test, y_pred, y_prob, cls_report, conf_matrix, best_params = train_and_evaluate_model(
                X, y, model_pipeline, model_type
            )
            
            # Display model performance
            st.markdown("#### Model Performance")
            st.write(f"Best Parameters: {best_params}")
            
            display_model_performance(model, X_test, y_test, y_pred, y_prob, cls_report, conf_matrix, 
                                     list(X.columns), model_type)
            
            # Display a success message
            st.success(f"Model training complete! The model has been saved as 'churn_prediction_{model_type}.pkl'.")
            
        else:
            # Check if a saved model exists
            model_path = f'churn_prediction_{model_type}.pkl'
            if os.path.exists(model_path):
                st.info(f"A saved {model_type} model exists. You can use it for predictions or retrain.")
            else:
                st.info("Click the 'Train Model' button to start training.")
    
    # Customer Prediction Tab
    with tab4:
        st.markdown("<div class='sub-header'>Customer Churn Prediction</div>", unsafe_allow_html=True)
        
        # Check if any models are available
        available_models = [f[:-4].split('_')[-1] for f in os.listdir('.') if f.startswith('churn_prediction_') and f.endswith('.pkl')]
        
        if not available_models:
            st.warning("No trained models found. Please go to the 'Predictive Modeling' tab and train a model first.")
        else:
            # Model selection
            selected_model = st.selectbox(
                "Select Model for Prediction", 
                options=available_models,
                format_func=lambda x: {
                    'random_forest': 'Random Forest',
                    'gradient_boosting': 'Gradient Boosting',
                    'xgboost': 'XGBoost',
                    'logistic_regression': 'Logistic Regression'
                }.get(x, x)
            )
            
            # Load the selected model
            model_path = f'churn_prediction_{selected_model}.pkl'
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                        st.success(f"Model '{selected_model}' loaded successfully.")       ####
                except Exception as e: 
                    st.error(f"Failed to load the model:{e}")
                    return 
                
                # Customer information input
                st.markdown("#### Enter Customer Information")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    gender = st.selectbox("Gender", options=['Male', 'Female'])
                    senior_citizen = st.selectbox("Senior Citizen", options=['No', 'Yes'])
                    partner = st.selectbox("Partner", options=['No', 'Yes'])
                    dependents = st.selectbox("Dependents", options=['No', 'Yes'])
                
                with col2:
                    tenure = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
                    phone_service = st.selectbox("Phone Service", options=['No', 'Yes'])
                    multiple_lines = st.selectbox(
                        "Multiple Lines", 
                        options=['No', 'Yes', 'No phone service'],
                        disabled=(phone_service == 'No')
                    )
                    internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
                
                with col3:
                    contract = st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year'])
                    paperless_billing = st.selectbox("Paperless Billing", options=['No', 'Yes'])
                    payment_method = st.selectbox(
                        "Payment Method", 
                        options=[
                            'Electronic check', 
                            'Mailed check', 
                            'Bank transfer (automatic)', 
                            'Credit card (automatic)'
                        ]
                    )
                    monthly_charges = st.slider("Monthly Charges ($)", min_value=0, max_value=150, value=70)
                
                # Additional services
                st.markdown("#### Additional Services")
                
                col1, col2, col3, col4 = st.columns(4)
                
                service_options = ['No', 'Yes', 'No internet service']
                
                with col1:
                    online_security = st.selectbox(
                        "Online Security", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                with col2:
                    online_backup = st.selectbox(
                        "Online Backup", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                with col3:
                    device_protection = st.selectbox(
                        "Device Protection", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                with col4:
                    tech_support = st.selectbox(
                        "Tech Support", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    streaming_tv = st.selectbox(
                        "Streaming TV", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                with col2:
                    streaming_movies = st.selectbox(
                        "Streaming Movies", 
                        options=service_options,
                        disabled=(internet_service == 'No')
                    )
                
                # Create customer data
                customer_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [monthly_charges * tenure]
                })
                
                # Create additional features
                customer_features = create_features(customer_data)
                
                # Make prediction button
                if st.button('Predict Churn Probability'):
                    # Make prediction
                    prediction, prediction_proba = customer_churn_prediction(model, customer_features)
                    
                    # Display prediction
                    st.markdown("#### Churn Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create a gauge chart for the churn probability
                        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
                        
                        # Define the colors based on the probability
                        if prediction_proba < 0.3:
                            color = 'green'
                        elif prediction_proba < 0.6:
                            color = 'orange'
                        else:
                            color = 'red'
                        
                        # Draw the gauge
                        ax.bar(x=np.pi/2, height=0.1, width=2*np.pi, bottom=0.8, color='lightgray', alpha=0.3)
                        ax.bar(x=np.pi/2, height=0.1, width=2*np.pi*prediction_proba, bottom=0.8, color=color)
                        
                        # Remove spines and ticks
                        ax.spines.clear()
                        ax.xaxis.set_visible(False)
                        ax.yaxis.set_visible(False)
                        
                        # Add text
                        ax.text(0, 0, f"{prediction_proba*100:.1f}%", fontsize=40, ha='center', va='center')
                        ax.text(0, -0.2, "Churn Probability", fontsize=16, ha='center', va='center')
                        
                        st.pyplot(fig)
                    
                    with col2:
                        # Get recommendations
                        risk_level, recommendations = get_churn_recommendations(df_processed, customer_features, prediction_proba)
                        
                        st.markdown(f"##### Risk Level: {risk_level}")
                        st.markdown("##### Recommendations:")
                        
                        for rec in recommendations:
                            st.markdown(f"- {rec}")
                    
                    # Show customer insights
                    st.markdown("#### Customer Insights")
                    visualize_customer_insights(df_processed, customer_features)

###
# ...existing code...
# ...existing code...

    # Add a floating "Back to Top" arrow button (anchor-based, works in Streamlit)
    st.markdown("""
        <style>
        #back-to-top-btn {
            position: fixed;
            bottom: 40px;
            right: 40px;
            z-index: 9999;
            background: #2563eb;
            color: white;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 32px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.85;
            text-decoration: none;
        }
        #back-to-top-btn:hover {
            background: #1e40af;
            opacity: 1;
        }
        </style>
        <a href="#top" id="back-to-top-btn" title="Back to Top">&#8679;</a>
    """, unsafe_allow_html=True)

    
# ...existing code...

# ...existing code...

    # Add a footer
    st.markdown("""
        <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #f1f1f1;
            text-align: center;
            padding: 10px;
        }
        </style>
        <div class="footer">
            <p>© 2025 Telecom Customer Churn Prediction. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

    # Add a button to clear the session state
    st.button("Clear Session State", on_click=lambda: st.session_state.clear(), key="clear_session_state")
     
# Run the application
if __name__ == "__main__":
    main() 