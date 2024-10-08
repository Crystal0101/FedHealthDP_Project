import pandas as pd
import yaml
import logging
import gc
import psutil
import os

# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载配置文件
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def load_data(file_name, usecols=None, dtype=None, chunksize=1000):
    logging.info(f"Loading data from {file_name}...")
    return pd.read_csv(
        config['data']['mimiciii_path'] + file_name,
        usecols=usecols,
        dtype=dtype,
        low_memory=True,
        chunksize=chunksize
    )

def log_memory_usage(stage):
    process = psutil.Process()
    mem_info = process.memory_info()
    logging.info(f"{stage} - Memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")

def merge_data_in_chunks(merged_data_path, chunk_generator, on_cols, suffix, left_on=None, right_on=None):
    merged_data = pd.read_csv(merged_data_path)
    for chunk in chunk_generator:
        logging.info(f"Merging chunk with columns: {chunk.columns.tolist()}")
        chunk = chunk.drop_duplicates()
        if left_on and right_on:
            merged_data = pd.merge(merged_data, chunk, left_on=left_on, right_on=right_on, how='left', suffixes=('', suffix))
        else:
            merged_data = pd.merge(merged_data, chunk, on=on_cols, how='left', suffixes=('', suffix))
        del chunk  # 释放内存
        gc.collect()  # 强制进行垃圾回收
        log_memory_usage("After merging chunk")
        # Write intermediate results to disk
        merged_data.to_csv(merged_data_path, index=False)
    return merged_data_path

def preprocess():
    try:
        logging.info("Starting preprocessing...")
        log_memory_usage("Initial")

        # 选择需要的列
        admissions_cols = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'insurance', 'ethnicity']
        patients_cols = ['subject_id', 'gender', 'dob']
        labevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']
        diagnoses_icd_cols = ['subject_id', 'hadm_id', 'icd9_code']
        prescriptions_cols = ['subject_id', 'hadm_id', 'startdate', 'drug']
        procedures_icd_cols = ['subject_id', 'hadm_id', 'icd9_code']
        condition_occurrence_cols = ['person_id', 'condition_concept_id', 'condition_start_date']
        observation_cols = ['person_id', 'observation_concept_id', 'observation_date']
        drug_exposure_cols = ['person_id', 'drug_concept_id', 'drug_exposure_start_date']
        chartevents_cols = ['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum']

        dtype_spec = {
            'insurance': str,
            'ethnicity': str,
            'valuenum': float
        }

        # 加载数据
        admissions_chunks = load_data("reduced_ADMISSIONS.csv", usecols=admissions_cols, dtype=dtype_spec)
        patients_chunks = load_data("reduced_PATIENTS.csv", usecols=patients_cols, dtype=dtype_spec)
        labevents_chunks = load_data("reduced_LABEVENTS.csv", usecols=labevents_cols, dtype=dtype_spec)
        diagnoses_icd_chunks = load_data("reduced_DIAGNOSES_ICD.csv", usecols=diagnoses_icd_cols, dtype=dtype_spec)
        prescriptions_chunks = load_data("reduced_PRESCRIPTIONS.csv", usecols=prescriptions_cols, dtype=dtype_spec)
        procedures_icd_chunks = load_data("reduced_PROCEDURES_ICD.csv", usecols=procedures_icd_cols, dtype=dtype_spec)
        condition_occurrence_chunks = load_data("reduced_CONDITION_OCCURRENCE.csv", usecols=condition_occurrence_cols, dtype=dtype_spec)
        observation_chunks = load_data("reduced_OBSERVATION.csv", usecols=observation_cols, dtype=dtype_spec)
        drug_exposure_chunks = load_data("reduced_DRUG_EXPOSURE.csv", usecols=drug_exposure_cols, dtype=dtype_spec)
        chartevents_chunks = load_data("reduced_CHARTEVENTS.csv", usecols=chartevents_cols, dtype=dtype_spec)

        # 处理数据块并合并
        admissions = pd.concat([chunk for chunk in admissions_chunks])
        patients = pd.concat([chunk for chunk in patients_chunks])
        
        # Clean the data
        admissions, patients = clean_data(admissions, patients)
        
        # Merge admissions and patients data
        merged_data = pd.merge(admissions, patients, on='subject_id', how='left')
        del admissions, patients  # 释放内存
        gc.collect()  # 强制进行垃圾回收
        log_memory_usage("After merging ADMISSIONS and PATIENTS")

        logging.info("Finished merging ADMISSIONS and PATIENTS.")

        # Save intermediate result
        merged_data_path = "merged_data.csv"
        merged_data.to_csv(merged_data_path, index=False)

        merged_data_path = merge_data_in_chunks(merged_data_path, labevents_chunks, ['subject_id', 'hadm_id'], '_labevents')
        merged_data_path = merge_data_in_chunks(merged_data_path, diagnoses_icd_chunks, ['subject_id', 'hadm_id'], '_diagnoses')
        merged_data_path = merge_data_in_chunks(merged_data_path, prescriptions_chunks, ['subject_id', 'hadm_id'], '_prescriptions')
        merged_data_path = merge_data_in_chunks(merged_data_path, procedures_icd_chunks, ['subject_id', 'hadm_id'], '_procedures')
        merged_data_path = merge_data_in_chunks(merged_data_path, condition_occurrence_chunks, ['subject_id'], '_conditions', left_on='subject_id', right_on='person_id')
        merged_data_path = merge_data_in_chunks(merged_data_path, observation_chunks, ['subject_id'], '_observations', left_on='subject_id', right_on='person_id')
        merged_data_path = merge_data_in_chunks(merged_data_path, drug_exposure_chunks, ['subject_id'], '_drug', left_on='subject_id', right_on='person_id')
        merged_data_path = merge_data_in_chunks(merged_data_path, chartevents_chunks, ['subject_id', 'hadm_id'], '_chart')

        logging.info(f"Final merged data path: {merged_data_path}")
        log_memory_usage("After merging all chunks")

        merged_data = pd.read_csv(merged_data_path)
        features, labels = extract_features(merged_data)

        logging.info("Preprocessing completed successfully.")
        
        return features, labels

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None, None

def clean_data(admissions, patients):
    logging.info("Cleaning data...")
    admissions.drop_duplicates(inplace=True)
    patients.drop_duplicates(inplace=True)

    admissions.fillna({'deathtime': '2199-01-01 00:00:00'}, inplace=True)
    patients.fillna('', inplace=True)

    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    admissions['dischtime'] = pd.to_datetime(admissions['dischtime'])
    admissions['deathtime'] = pd.to_datetime(admissions['deathtime'])
    patients['dob'] = pd.to_datetime(patients['dob'])

    logging.info("Data cleaned.")
    return admissions, patients

def extract_features(merged_data):
    logging.info("Extracting features...")

    merged_data['length_of_stay'] = (merged_data['dischtime'] - merged_data['admittime']).dt.days
    merged_data['age'] = merged_data['admittime'].dt.year - merged_data['dob'].dt.year
    merged_data['is_dead'] = merged_data['deathtime'].apply(lambda x: 0 if x == pd.Timestamp('2199-01-01 00:00:00') else 1)

    features = merged_data[['length_of_stay', 'age', 'is_dead', 'admission_type', 'insurance', 'ethnicity', 'valuenum']]

    features.fillna({'valuenum': 0}, inplace=True)

    num_features = features.select_dtypes(include=[int, float])
    num_features.fillna(num_features.mean(), inplace=True)
    cat_features = features.select_dtypes(include=[object])
    cat_features.fillna('Unknown', inplace=True)

    features = pd.concat([num_features, cat_features], axis=1)
    features = pd.get_dummies(features, columns=['admission_type', 'insurance', 'ethnicity'], drop_first=True)

    diagnosis_counts = merged_data['icd9_code'].value_counts()
    logging.info(f"Diagnosis counts: {diagnosis_counts}")

    min_freq = 10
    common_diagnoses = diagnosis_counts[diagnosis_counts >= min_freq].index
    logging.info(f"Common diagnoses: {common_diagnoses}")

    merged_data['icd9_code'] = merged_data['icd9_code'].apply(lambda x: x if x in common_diagnoses else 'Other')
    labels = merged_data['icd9_code'].astype('category').cat.codes

    logging.info("Feature extraction completed.")

    return features, labels

# 示例调用
if __name__ == "__main__":
    features, labels = preprocess()
    if features is not None and labels is not None:
        print(features.head(), labels.head())
    else:
        print("Preprocessing failed.")
