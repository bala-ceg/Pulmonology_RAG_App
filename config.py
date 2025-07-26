import os

BASE_STORAGE_PATH = "./KB/"
VECTOR_DB_PATH = "./vector_dbs/"
VECTOR_DBS_FOLDER = VECTOR_DB_PATH

os.makedirs(VECTOR_DB_PATH, exist_ok=True)
