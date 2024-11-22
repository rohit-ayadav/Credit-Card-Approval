import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import joblib
from pathlib import Path

class CreditApplicationProcessor:
    def __init__(self, model_path: str = "credit_model.joblib"):
        """Initialize the credit application processor with model and preprocessing objects."""
        self.model_path = Path(model_path)
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = ['Gender', 'Industry', 'Ethnicity', 'Citizen']
        self.numeric_columns = ['Age', 'Debt', 'YearsEmployed', 'CreditScore', 'Income']
        self.binary_columns = ['Married', 'BankCustomer', 'PriorDefault', 'Employed', 
                             'DriversLicense']
        
    def load_or_train_model(self, training_data_path: str = None) -> None:
        """Load existing model or train a new one if needed."""
        try:
            if self.model_path.exists():
                saved_objects = joblib.load(self.model_path)
                self.model = saved_objects['model']
                self.scaler = saved_objects['scaler']
                self.label_encoders = saved_objects['label_encoders']
                print("Loaded existing model and preprocessing objects.")
            elif training_data_path:
                self._train_new_model(training_data_path)
            else:
                raise FileNotFoundError("No model found and no training data provided.")
        except Exception as e:
            raise RuntimeError(f"Error loading/training model: {str(e)}")

    def _train_new_model(self, training_data_path: str) -> None:
        """Train a new model using the provided training data."""
        try:
            # Load and preprocess training data
            train_data = pd.read_csv(training_data_path)
            X_train = self._preprocess_data(train_data.drop('Approved', axis=1))
            y_train = train_data['Approved']
            
            # Initialize and train model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            self.model.fit(X_train, y_train)
            
            # Save model and preprocessing objects
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders
            }, self.model_path)
            
            print("Trained and saved new model.")
        except Exception as e:
            raise RuntimeError(f"Error training new model: {str(e)}")

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input data for model prediction."""
        processed_data = data.copy()
        
        # Handle categorical variables
        for col in self.categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            processed_data[col] = self.label_encoders[col].fit_transform(processed_data[col])
        
        # Scale numeric variables
        numeric_data = processed_data[self.numeric_columns]
        processed_data[self.numeric_columns] = self.scaler.fit_transform(numeric_data)
        
        # Convert binary columns to int
        for col in self.binary_columns:
            processed_data[col] = processed_data[col].astype(int)
            
        return processed_data
    def analyze_applications(self, test_data_path: str) -> None:
        """Analyze existing credit card applications with visualizations."""
        try:
            test_data = pd.read_csv(test_data_path)
            total_applications = len(test_data)
            approved = len(test_data[test_data['Approved'] == 1])
            rejected = total_applications - approved
            
            print("\n=== Credit Card Application Analysis ===")
            print(f"Total Applications: {total_applications:,}")
            print(f"Approved Applications: {approved:,} ({(approved/total_applications)*100:.2f}%)")
            print(f"Rejected Applications: {rejected:,} ({(rejected/total_applications)*100:.2f}%)")
            
            self._create_visualizations(test_data)
            
        except Exception as e:
            print(f"Error analyzing applications: {str(e)}")


    def _create_visualizations(self, data: pd.DataFrame) -> None:
        """Create detailed visualizations for application analysis."""
        plt.style.use('fivethirtyeight')  # Using a built-in style
        fig = plt.figure(figsize=(20, 10))
        
        # Set consistent colors for approved/rejected
        colors = ['#ff9999', '#66b3ff']
        
        # Approval Distribution
        plt.subplot(2, 3, 1)
        approval_counts = data['Approved'].value_counts()
        plt.pie(approval_counts, labels=['Rejected', 'Approved'], 
               autopct='%1.1f%%', colors=colors)
        plt.title('Application Distribution', pad=20)
        
        # Age Distribution
        plt.subplot(2, 3, 2)
        sns.boxplot(x='Approved', y='Age', data=data, hue='Approved', palette=colors, legend=False)
        plt.title('Age Distribution by Approval Status', pad=20)
        plt.xlabel('Application Status (0=Rejected, 1=Approved)')
        
        # Income Distribution
        plt.subplot(2, 3, 3)
        sns.boxplot(x='Approved', y='Income', data=data, palette=colors, hue='Approved', legend=False)
        plt.title('Income Distribution by Approval Status', pad=20)
        plt.xlabel('Application Status (0=Rejected, 1=Approved)')
        
        # Credit Score Distribution
        plt.subplot(2, 3, 4)
        sns.histplot(data=data, x='CreditScore', hue='Approved', multiple="stack",
                    palette=colors, kde=True)
        plt.title('Credit Score Distribution', pad=20)
        plt.xlabel('Credit Score')
        
        # Years Employed Distribution
        plt.subplot(2, 3, 5)
        sns.boxplot(x='Approved', y='YearsEmployed', data=data, palette=colors, hue='Approved', legend=False)
        plt.title('Years Employed Distribution', pad=20)
        plt.xlabel('Application Status (0=Rejected, 1=Approved)')
        
        # Debt Distribution
        plt.subplot(2, 3, 6)
        sns.boxplot(x='Approved', y='Debt', data=data, palette=colors, hue='Approved', legend=False)
        plt.title('Debt Distribution', pad=20)
        plt.xlabel('Application Status (0=Rejected, 1=Approved)')
        
        # Adjust layout and display
        plt.tight_layout(pad=3.0)
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plots: {str(e)}")
    def get_user_application(self) -> Dict[str, Any]:
        """Interactive function to get and validate user application details."""
        def validate_choice(prompt: str, choices: list) -> str:
            while True:
                value = input(prompt).strip().upper()
                if value in choices:
                    return value
                print(f"Invalid input! Please enter one of: {', '.join(choices)}")

        def validate_numeric(prompt: str, value_type: type = float, 
                           minimum: float = 0, maximum: float = float('inf')) -> float:
            while True:
                try:
                    value = value_type(input(prompt).strip())
                    if minimum <= value <= maximum:
                        return value
                    print(f"Please enter a value between {minimum} and {maximum}.")
                except ValueError:
                    print(f"Invalid input! Please enter a valid {value_type.__name__}.")

        print("\n=== New Credit Card Application ===")
        
        application = {
            'Gender': validate_choice("Enter Gender (M/F): ", ['M', 'F']),
            'Age': validate_numeric("Enter Age: ", int, 18, 120),
            'Debt': validate_numeric("Enter Debt amount: ", float, 0, 1e7),
            'Married': validate_choice("Are you Married? (1/0): ", ['1', '0']),
            'BankCustomer': validate_choice("Are you a Bank Customer? (1/0): ", ['1', '0']),
            'Industry': input("Enter Industry: ").strip(),
            'Ethnicity': input("Enter Ethnicity: ").strip(),
            'YearsEmployed': validate_numeric("Years Employed: ", float, 0, 60),
            'PriorDefault': validate_choice("Any Prior Default? (1/0): ", ['1', '0']),
            'Employed': validate_choice("Currently Employed? (1/0): ", ['1', '0']),
            'CreditScore': validate_numeric("Enter Credit Score: ", int, 300, 850),
            'DriversLicense': validate_choice("Have Drivers License? (1/0): ", ['1', '0']),
            'Citizen': input("Citizenship Status: ").strip(),
            'ZipCode': validate_numeric("Enter ZipCode: ", int, 0, 99999),
            'Income': validate_numeric("Enter Annual Income: ", float, 0, 1e7)
        }
        
        return application

    def predict_application(self, application: Dict[str, Any]) -> Tuple[int, float]:
        """Predict the outcome of a single application with probability."""
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded. Please load or train the model first.")
                
            # Convert application to DataFrame and preprocess
            application_df = pd.DataFrame([application])
            processed_application = self._preprocess_data(application_df)
            
            # Make prediction
            prediction = self.model.predict(processed_application)[0]
            probability = self.model.predict_proba(processed_application)[0][1]
            
            return prediction, probability
            
        except Exception as e:
            raise RuntimeError(f"Error predicting application: {str(e)}")

def main():
    # Initialize processor
    processor = CreditApplicationProcessor()
    
    try:
        # Load or train model
        processor.load_or_train_model(training_data_path="training_data.csv")
        
        # Analyze existing applications
        processor.analyze_applications("test_data.csv")
        
        # Interactive application processing
        while True:
            choice = input("\nWould you like to submit a new application? (yes/no): ").lower()
            if choice not in ['yes', 'y']:
                break
                
            try:
                application = processor.get_user_application()
                prediction, probability = processor.predict_application(application)
                
                print("\n=== Application Results ===")
                print(f"Status: {'Approved' if prediction == 1 else 'Rejected'}")
                print(f"Approval Probability: {probability:.2%}")
                
            except Exception as e:
                print(f"Error processing application: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        return
    print("\nThank you for using the credit card application processor")
    print("Goodbye!")

if __name__ == "__main__":
    main()