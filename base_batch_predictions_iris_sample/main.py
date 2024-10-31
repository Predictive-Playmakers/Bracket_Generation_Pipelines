import json
import yaml
import pymysql
import functions_framework
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
from flask import jsonify

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract database and Vertex AI configurations
INSTANCE_CONNECTION_NAME = config["database"]["instance_connection_name"]
DB_USER = config["database"]["user"]
# DB_PASS = config["database"]["password"]
DB_NAME = config["database"]["name"]

PROJECT_ID = config["vertex_ai"]["project_id"]
REGION = config["vertex_ai"]["region"]
ENDPOINT_ID = config["vertex_ai"]["endpoint_id"]


# Initialize database connector
connector = Connector()


def get_db_connection():
    # Securely connect to MySQL database using Cloud SQL connector
    return connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pymysql",
        user=DB_USER,
        # password=DB_PASS,
        db=DB_NAME,
    )


@functions_framework.http
def predict_iris(request):
    """HTTP Cloud Function to process iris sample IDs, call Vertex AI, and store results."""
    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "iris_ids" not in request_json:
        return "Invalid input: 'iris_ids' list required", 400

    iris_ids = request_json["iris_ids"]

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

        # Step 2: Call Vertex AI endpoint with batch of iris features
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

        predictions = endpoint.predict(instances=feature_rows).predictions

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
