import json
import yaml
import pymysql
import functions_framework
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
from flask import jsonify

with open("env.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract database and Vertex AI configurations
INSTANCE_CONNECTION_NAME = config["INSTANCE_CONNECTION_NAME"]
DB_USER = config["DB_USER"]
DB_PASS = config["DB_PASS"]
DB_NAME = config["DB_NAME"]

PROJECT_ID = config["PROJECT_ID"]
REGION = config["REGION"]
ENDPOINT_ID = config["ENDPOINT_ID"]


def get_db_connection(connector):
    try:
        # Initialize database connector
        connector = Connector()

        connection = connector.connect(
            INSTANCE_CONNECTION_NAME,
            "pymysql",
            user=DB_USER,
            password=DB_PASS,
            db=DB_NAME,
        )
        return connection
    except Exception as e:
        print("Failed to connect to Cloud SQL:", e)
        raise


@functions_framework.http
def predict_iris(request):
    """HTTP Cloud Function to process iris sample IDs, call Vertex AI, and store results."""
    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "iris_ids" not in request_json:
        return "Invalid input: 'iris_ids' list required", 400

    iris_ids = request_json["iris_ids"]

    print("Iris IDs:", iris_ids)
    try:
        # Step 1: Connect to Cloud SQL to fetch iris sample features
        conn = get_db_connection()
        cursor = conn.cursor()
        feature_rows = []

        # Fetch features for each iris sample ID
        for iris_id in iris_ids:
            cursor.execute(
                "SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_samples WHERE id = %s",
                (iris_id,),
            )
            iris_features = cursor.fetchone()

            # Append features to feature_rows if they exist
            if iris_features:
                feature_rows.append(list(iris_features))

        print("Fetched iris features")
        print("Feature rows:", feature_rows)

        # Step 2: Call Vertex AI endpoint with batch of iris features
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
        print("Initialized Vertex AI endpoint")

        predictions = endpoint.predict(instances=feature_rows).predictions
        print("Received predictions from Vertex AI")
        print("Predictions:", predictions)

        # Step 3v1: Prepare predictions for JSON response
        results_data = [
            {"id": iris_id, "prediction": pred}
            for iris_id, pred in zip(iris_ids, predictions)
        ]

        # Return predictions as JSON response
        return jsonify({"predictions": results_data}), 200

        # # Step 3v2: Store predictions in MySQL `iris_results` table
        # results_data = [(iris_id, pred) for iris_id, pred in zip(iris_ids, predictions)]

        # insert_query = """
        #     INSERT INTO iris_results (id, prediction)
        #     VALUES (%s, %s)
        # """

        # cursor.executemany(insert_query, results_data)
        # conn.commit()  # Commit transaction

        # # Close database connection
        # cursor.close()
        # conn.close()

        # # Return a success response
        # return jsonify({"status": "Predictions stored successfully"}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "Error", "details": str(e)}), 500

    finally:
        # Clean up connector to avoid open connections
        connector.close()
