from google.cloud import storage
from google.cloud import bigquery
import sqlite3
import pandas as pd

BUCKET_NAME = "usc-data" #Change as per your setup
OBJECT_NAME = "edit-pathways/newssniffer-nytimes.db" #Change as per your setup
DATABASE_NAME_IN_RUNTIME = "/tmp/nytimes.db" #Remember that only the /tmp folder is writable within the directory
QUERY = "SELECT * FROM artists;" #Change as per your query
TABLE_ID = "usc-research.news_edits.news_edits_entryversion" # Change with the format your-project.your_dataset.your_table_name

storage_client = storage.Client()
bigquery_client = bigquery.Client()


# Fetch the .db file from Cloud Storage
def get_db_file_from_gcs(bucket_name, object_name, filename):
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    return blob.download_to_filename(filename)


#Makes the connections to the DB
def connect_to_sqlite_db(complete_filepath_to_db):
    connection = sqlite3.connect(complete_filepath_to_db)
    return connection


#Run the query and return a cursor to iterate over the results
def run_query_to_db(connection, query):
    with connection:
        cursor = connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()


#Run the query and saves it to a dataframe
def run_query_to_db_with_pandas(connection, query):
    with connection:
        df = pd.read_sql_query(query, connection)
        return df


def gcssqlite_to_bq(request):
    print("Getting .db file from Storage")
    get_db_file_from_gcs(BUCKET_NAME, OBJECT_NAME, DATABASE_NAME_IN_RUNTIME)
    print("Downloaded .db file in CF instance RAM")
    print("Trying to connect to database using sqlite")
    cnx = connect_to_sqlite_db(DATABASE_NAME_IN_RUNTIME)
    print("Connected to database")
    print("Attempting to perform a query")
    results = run_query_to_db_with_pandas(cnx, QUERY)
    print("Writing data to BigQuery")
    bigqueryJob = bigquery_client.load_table_from_dataframe(results, TABLE_ID)
    bigqueryJob.result()
    print("The Job to write to Big Query is finished")
    return "Executed Function"


# gcloud functions deploy test \
#   --region us-central1 \
#   --entry-point gcssqlite_to_bq \
#   --timeout 540 \
#   --memory 1024MB \
#   --runtime python38 \
#   --trigger-http \
#   --allow-unauthenticated