import os
from dask import dataframe as dd
import pandas as pd


mimic_dir = 'MIMICIII/physionet.org/files/mimiciii/1.4/'

# root, dirs, files = next(os.walk(mimic_dir))

# f = open('temp_mimic.txt', 'w')
# f2 = open('temp_mimic2.txt', 'w')

# for file in files:
#     # print(file)
#     split = file.split('.')
#     # print(split)
#     if len(split) == 2 and split[1] == 'csv':
#         print(split)
#         name = split[0]
#         f.write(f"{name.lower()}_path = f'{{mimic_dir}}{name}.csv'\n")
#         f2.write(f"{name.lower()} = dd.read_csv({name.lower()}_path)\n")

# f.close()
# f2.close()
# exit()


procedureevents_mv_path = f'{mimic_dir}PROCEDUREEVENTS_MV.csv'
callout_path = f'{mimic_dir}CALLOUT.csv'
d_cpt_path = f'{mimic_dir}D_CPT.csv'
d_items_path = f'{mimic_dir}D_ITEMS.csv'
caregivers_path = f'{mimic_dir}CAREGIVERS.csv'
microbiologyevents_path = f'{mimic_dir}MICROBIOLOGYEVENTS.csv'
labevents_path = f'{mimic_dir}LABEVENTS.csv'
inputevents_cv_path = f'{mimic_dir}INPUTEVENTS_CV.csv'
admissions_path = f'{mimic_dir}ADMISSIONS.csv'
d_labitems_path = f'{mimic_dir}D_LABITEMS.csv'
datetimeevents_path = f'{mimic_dir}DATETIMEEVENTS.csv'
prescriptions_path = f'{mimic_dir}PRESCRIPTIONS.csv'
procedures_icd_path = f'{mimic_dir}PROCEDURES_ICD.csv'
noteevents_path = f'{mimic_dir}NOTEEVENTS.csv'
chartevents_path = f'{mimic_dir}CHARTEVENTS.csv'
transfers_path = f'{mimic_dir}TRANSFERS.csv'
diagnoses_icd_path = f'{mimic_dir}DIAGNOSES_ICD.csv'
services_path = f'{mimic_dir}SERVICES.csv'
drgcodes_path = f'{mimic_dir}DRGCODES.csv'
outputevents_path = f'{mimic_dir}OUTPUTEVENTS.csv'
patients_path = f'{mimic_dir}PATIENTS.csv'
d_icd_diagnoses_path = f'{mimic_dir}D_ICD_DIAGNOSES.csv'
icustays_path = f'{mimic_dir}ICUSTAYS.csv'
inputevents_mv_path = f'{mimic_dir}INPUTEVENTS_MV.csv'
d_icd_procedures_path = f'{mimic_dir}D_ICD_PROCEDURES.csv'
cptevents_path = f'{mimic_dir}CPTEVENTS.csv'


# procedureevents_mv = dd.read_csv(procedureevents_mv_path).set_index('ROW_ID')
# callout = dd.read_csv(callout_path, assume_missing=True).set_index('ROW_ID')
# d_cpt = dd.read_csv(d_cpt_path).set_index('ROW_ID')
# d_items = dd.read_csv(d_items_path, dtype={
#     'ABBREVIATION': 'object',
#     'PARAM_TYPE': 'object',
#     'UNITNAME': 'object',
# }).set_index('ROW_ID')
# caregivers = dd.read_csv(caregivers_path).set_index('ROW_ID')
# microbiologyevents = dd.read_csv(microbiologyevents_path, dtype={
#     'SPEC_ITEMID': 'float64',
# }).set_index('ROW_ID')
# labevents = dd.read_csv(labevents_path).set_index('ROW_ID')
# inputevents_cv = dd.read_csv(inputevents_cv_path).set_index('ROW_ID')
admissions = dd.read_csv(admissions_path).set_index('ROW_ID')
# d_labitems = dd.read_csv(d_labitems_path).set_index('ROW_ID')
# datetimeevents = dd.read_csv(datetimeevents_path).set_index('ROW_ID')
# prescriptions = dd.read_csv(prescriptions_path).set_index('ROW_ID')
# procedures_icd = dd.read_csv(procedures_icd_path).set_index('ROW_ID')
# noteevents = dd.read_csv(noteevents_path).set_index('ROW_ID')
# chartevents = dd.read_csv(chartevents_path).set_index('ROW_ID')
# transfers = dd.read_csv(transfers_path).set_index('ROW_ID')
diagnoses_icd = dd.read_csv(diagnoses_icd_path, assume_missing=True).set_index('ROW_ID')
# services = dd.read_csv(services_path).set_index('ROW_ID')
# drgcodes = dd.read_csv(drgcodes_path).set_index('ROW_ID')
# outputevents = dd.read_csv(outputevents_path).set_index('ROW_ID')
patients = dd.read_csv(patients_path).set_index('ROW_ID')
d_icd_diagnoses = dd.read_csv(d_icd_diagnoses_path, dtype={'ICD9_CODE': 'object'}).set_index('ROW_ID')
# icustays = dd.read_csv(icustays_path).set_index('ROW_ID')
# inputevents_mv = dd.read_csv(inputevents_mv_path).set_index('ROW_ID')
# d_icd_procedures = dd.read_csv(d_icd_procedures_path).set_index('ROW_ID')
# cptevents = dd.read_csv(cptevents_path).set_index('ROW_ID')

# HADM_ID is duplicate
# diagnoses_icd = diagnoses_icd.drop(['HADM_ID'], axis=1)
# print(diagnoses_icd.compute())
# diagnoses_icd_full = diagnoses_icd.merge(d_icd_diagnoses, how='inner', on='ICD9_CODE')
# print(diagnoses_icd_full.compute())
# exit()

# print(admissions.compute())
patients_diagnosis = patients.merge(diagnoses_icd.drop(['SEQ_NUM'], axis=1), how='left', on='SUBJECT_ID')
idx = patients.set_index('SUBJECT_ID').index.compute()
print(idx)
# print(patients_diagnosis.compute())

# merged = merged.merge(admissions.drop(['SUBJECT_ID'], axis=1), how='inner', on='HADM_ID')
# print(merged.compute())

septic_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78552']}), npartitions=1)
hemo_shock = dd.from_pandas(pd.DataFrame({'ICD9_CODE': ['78559', '99809', '9584']}), npartitions=1)
# print(hemo_shock.compute())
# selected_disease = selected_disease.merge(d_icd_diagnoses, how='inner', on='ICD9_CODE')

# print(selected_disease.compute())

# septic shock
positives_septic = patients_diagnosis.merge(septic_shock, how='inner', on='ICD9_CODE')#.set_index('SUBJECT_ID')
positives_septic = positives_septic.drop_duplicates(subset=['SUBJECT_ID']).set_index('SUBJECT_ID').index
positives_septic_idx = positives_septic.compute()

negatives_septic_idx = idx.difference(positives_septic_idx)
print(negatives_septic_idx)

positives = pd.DataFrame({'y': 1}, index=positives_septic_idx)
negatives = pd.DataFrame({'y': 0}, index=negatives_septic_idx)
tot = pd.concat((positives, negatives), axis=0).sort_index()
print(positives)
print(negatives)
print(tot)

exit()

# hemo shock
# positives_hemo = patients_diagnosis.merge(hemo_shock, how='inner', on='ICD9_CODE')#.set_index('SUBJECT_ID')
# positives_hemo = positives_hemo.drop_duplicates(subset=['SUBJECT_ID']).set_index('SUBJECT_ID').index
# print(positives_hemo.compute())

# print(positives_hemo.drop_duplicates(subset=['SUBJECT_ID']).compute())
# positives_hemo = positives_hemo.compute().set_index('SUBJECT_ID')#.index
# print(positives_hemo)

# print(positives_septic)
# print(positives_hemo)
exit()
# # positives = positives.compute().filter(['y', 'SUBJECT_ID'], axis=1)
# # positives.set_index('SUBJECT_ID')
# positives = positives.compute().filter(['SUBJECT_ID'], axis=1).set_index('SUBJECT_ID')
# # controls = patients.set_index('SUBJECT_ID').compute().drop(positives.index, axis=0)
# controls = patients.set_index()
# # positives.filter(['y', 'SUBJECT_ID'], axis=1)
# print(positives)
# print(controls)
