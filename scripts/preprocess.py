import pandas as pd
import logging
import gc
import yaml
import os
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration file
with open("/content/FedHealthDP_Project/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_data(file_name, usecols=None, dtype=None, chunksize=50000):
    file_path = os.path.join(config['data']['mimiciii_path'], file_name)
    logging.info(f"Loading data from {file_path}...")
    try:
        data_iter = pd.read_csv(
            file_path,
            usecols=usecols,
            dtype=dtype,
            low_memory=True,
            chunksize=chunksize
        )
        data = pd.concat(data_iter)
        gc.collect()  # Explicitly free memory
        return data
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return None

def preprocess():
    try:
        logging.info("Starting preprocessing...")

        # Define columns to load
        patients_cols = ['subject_id', 'dob']
        admissions_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime']
        diagnoses_icd_cols = ['subject_id', 'hadm_id', 'icd9_code']
        labevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
        chartevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'value']
        prescriptions_cols = ['subject_id', 'hadm_id', 'startdate', 'drug']

        # Define data types for specific columns
        dtype_spec = {'icd9_code': str, 'valuenum': float, 'value': str, 'drug': str}

        # Load datasets
        patients = load_data("PATIENTS.csv", usecols=patients_cols, dtype=dtype_spec)
        admissions = load_data("ADMISSIONS.csv", usecols=admissions_cols, dtype=dtype_spec)
        diagnoses_icd = load_data("DIAGNOSES_ICD.csv", usecols=diagnoses_icd_cols, dtype=dtype_spec)
        labevents = load_data("LABEVENTS.csv", usecols=labevents_cols, dtype=dtype_spec)
        chartevents = load_data("CHARTEVENTS.csv", usecols=chartevents_cols, dtype=dtype_spec)
        prescriptions = load_data("PRESCRIPTIONS.csv", usecols=prescriptions_cols, dtype=dtype_spec)

        if any(df is None for df in [patients, admissions, diagnoses_icd, labevents, chartevents, prescriptions]):
            logging.error("One or more data files could not be loaded.")
            return None, None

        # Convert date columns to datetime
        patients['dob'] = pd.to_datetime(patients['dob'], errors='coerce')
        admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
        admissions['dischtime'] = pd.to_datetime(admissions['dischtime'], errors='coerce')
        labevents['charttime'] = pd.to_datetime(labevents['charttime'], errors='coerce')
        chartevents['charttime'] = pd.to_datetime(chartevents['charttime'], errors='coerce')
        prescriptions['startdate'] = pd.to_datetime(prescriptions['startdate'], errors='coerce')

        # Calculate age and length of stay
        admissions['age'] = admissions['admittime'].dt.year - patients['dob'].dt.year
        admissions['length_of_stay'] = (admissions['dischtime'] - admissions['admittime']).dt.days

        # Merge datasets
        merged_data = admissions.merge(patients[['subject_id']], on='subject_id', how='left')
        merged_data = merged_data.merge(diagnoses_icd, on=['subject_id', 'hadm_id'], how='left')
        merged_data = merged_data.merge(labevents, on=['subject_id', 'hadm_id'], how='left')
        merged_data = merged_data.merge(chartevents, on=['subject_id', 'hadm_id'], how='left')
        merged_data = merged_data.merge(prescriptions, on=['subject_id', 'hadm_id'], how='left')

        # Filter cancer-related ICD-9 codes
        cancer_icd_codes = ['140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169']
        merged_data['is_cancer'] = merged_data['icd9_code'].apply(lambda x: 1 if str(x)[:3] in cancer_icd_codes else 0)

        # Select and aggregate features
        merged_data['value'] = pd.to_numeric(merged_data['value'], errors='coerce')
        features = merged_data[['subject_id', 'age', 'length_of_stay', 'valuenum', 'value']]
        features.fillna(0, inplace=True)
        features = features.groupby('subject_id').mean().reset_index()

        # Prepare labels
        labels = merged_data[['subject_id', 'is_cancer']].drop_duplicates()
        labels = labels.groupby('subject_id').max().reset_index()

        # Ensure consistency between features and labels
        combined = features.merge(labels, on='subject_id', how='inner')
        features = combined.drop(columns=['is_cancer'])
        labels = combined['is_cancer']

        # Split data for federated learning (simulating different clients)
        num_clients = 5  # Define the number of clients
        client_data = []

        for i in range(num_clients):
            client_features, _, client_labels, _ = train_test_split(features, labels, test_size=0.8, random_state=i)
            client_data.append((client_features, client_labels))

        return client_data

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


# 示例调用
if __name__ == "__main__":
    features, labels = preprocess()
    if features is not None and labels is not None:
        print(features.head(), labels.head())
    else:
        print("Preprocessing failed.")
