import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_credit_applications():
    # Read test data
    test_data = pd.read_csv("test_data.csv")
    
    # Basic statistics
    total_applications = len(test_data)
    approved = len(test_data[test_data['Approved'] == 1])
    rejected = len(test_data[test_data['Approved'] == 0])
    
    print("\n=== Credit Card Application Analysis ===")
    print(f"\nTotal Applications: {total_applications}")
    print(f"Approved Applications: {approved} ({(approved/total_applications)*100:.2f}%)")
    print(f"Rejected Applications: {rejected} ({(rejected/total_applications)*100:.2f}%)")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    # Approval Distribution
    plt.subplot(1, 3, 1)
    plt.pie([approved, rejected], labels=['Approved', 'Rejected'], autopct='%1.1f%%')
    plt.title('Application Distribution')
    
    # Age Distribution
    plt.subplot(1, 3, 2)
    sns.boxplot(x='Approved', y='Age', data=test_data)
    plt.title('Age Distribution by Approval Status')
    
    # Income Distribution
    plt.subplot(1, 3, 3)
    sns.boxplot(x='Approved', y='Income', data=test_data)
    plt.title('Income Distribution by Approval Status')
    
    plt.tight_layout()
    plt.show()

def get_user_application():
    """Interactive function to get and validate user application details."""
    print("\n=== New Credit Card Application ===")
    
    application = {}
    
    # Validation functions for specific input fields
    def validate_choice(prompt, choices):
        """Validate user input based on predefined choices."""
        while True:
            value = input(prompt).strip().upper()
            if value in choices:
                return value
            print(f"Invalid input! Please enter one of the following: {', '.join(choices)}")

    def validate_numeric(prompt, value_type=float, positive=False):
        """Validate numeric input based on the type and constraints."""
        while True:
            try:
                value = value_type(input(prompt).strip())
                if positive and value < 0:
                    raise ValueError
                return value
            except ValueError:
                print(f"Invalid input! Please enter a {'positive ' if positive else ''}{value_type.__name__}.")

    # Collect and validate user inputs
    application['Gender'] = validate_choice("Enter Gender (M/F): ", ['M', 'F'])
    application['Age'] = validate_numeric("Enter Age: ", float, positive=True)
    application['Debt'] = validate_numeric("Enter Debt amount: ", float, positive=True)
    application['Married'] = validate_choice("Are you Married? (1 for Yes, 0 for No): ", ['1', '0'])
    application['BankCustomer'] = validate_choice("Are you a Bank Customer? (1 for Yes, 0 for No): ", ['1', '0'])
    application['Industry'] = input("Enter Industry (e.g., Healthcare, Research, Technology): ").strip()
    application['Ethnicity'] = input("Enter Ethnicity: ").strip()
    application['YearsEmployed'] = validate_numeric("Years Employed: ", float, positive=True)
    application['PriorDefault'] = validate_choice("Any Prior Default? (1 for Yes, 0 for No): ", ['1', '0'])
    application['Employed'] = validate_choice("Currently Employed? (1 for Yes, 0 for No): ", ['1', '0'])
    application['CreditScore'] = validate_numeric("Enter Credit Score: ", int, positive=True)
    application['DriversLicense'] = validate_choice("Have Drivers License? (1 for Yes, 0 for No): ", ['1', '0'])
    application['Citizen'] = input("Citizenship Status (e.g., ByBirth, Naturalized): ").strip()
    application['ZipCode'] = validate_numeric("Enter ZipCode: ", int, positive=True)
    application['Income'] = validate_numeric("Enter Annual Income: ", float, positive=True)
    
    return application

def predict_application(application, model):
    """Predict the outcome of a single application"""
    # Convert application to DataFrame
    application_df = pd.DataFrame([application])

    prediction = model.predict(application_df)
    probability = model.predict_proba(application_df)
    
    return prediction[0], probability[0][1]

def main():
    # Analyze existing applications
    analyze_credit_applications()
    
    # Interactive application processing
    while True:
        choice = input("\nWould you like to submit a new application? (yes/no): ")
        if choice.lower() not in ['yes', 'y']:
            break
            
        application = get_user_application()
        if application:
            # Load the trained model
            model = RandomForestClassifier()
            model.load("credit_approval_model.pkl")
            
            # Predict application outcome
            prediction, probability = predict_application(application, model)
            
            # Display result
            result = "Approved" if prediction == 1 else "Rejected"
            print(f"\nApplication Result: {result}")
            print(f"Probability of Approval: {probability:.2f}")
if __name__ == "__main__":
    main()