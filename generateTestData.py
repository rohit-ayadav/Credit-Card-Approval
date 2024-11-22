import pandas as pd
import random

# Read data from the credit card data file
def read_credit_card_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

# Generate test data by sampling random rows and adding an 'approved' field if needed
def generate_test_data(data, output_file, sample_size=100):
    if data is not None:
        try:
            # Ensure the sample size does not exceed available rows
            sample_size = min(sample_size, len(data))
            
            # Randomly sample rows
            test_data = data.sample(n=sample_size, random_state=42)
            
            # Ensure the 'approved' field exists, or add it if missing
            if 'approved' not in test_data.columns:
                test_data['approved'] = [random.choice([0, 1]) for _ in range(len(test_data))]
            
            # Save test data to a new file
            test_data.to_csv(output_file, index=False)
            print(f"Test data generated and saved to {output_file}")
        except Exception as e:
            print(f"An error occurred while generating test data: {e}")
    else:
        print("No data to generate test data from.")

# Main function
if __name__ == "__main__":
    input_file = "credit_card_data.csv"  # Input file path
    output_file = "test_data.csv"        # Output file path
    
    # Read the data
    credit_card_data = read_credit_card_data(input_file)
    
    # Generate test data
    generate_test_data(credit_card_data, output_file)
