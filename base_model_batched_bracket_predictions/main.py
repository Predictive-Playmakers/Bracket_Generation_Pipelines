import json
import pymysql
import yaml
import functions_framework
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
from flask import jsonify

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract database and Vertex AI configurations
INSTANCE_CONNECTION_NAME = config["database"]["instance_connection_name"]
DB_USER = config["database"]["user"]
DB_PASS = config["database"]["password"]
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
        password=DB_PASS,
        db=DB_NAME,
    )


@functions_framework.http
def predict_matchups(request):
    """HTTP Cloud Function to process matchups, call Vertex AI, and store results."""
    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "matchups" not in request_json:
        return "Invalid input: 'matchups' data required", 400

    matchups = request_json["matchups"]

    try:
        # Step 1: Connect to Cloud SQL to fetch team features
        conn = get_db_connection()
        cursor = conn.cursor()

        feature_rows = []
        pair_ids = []  # Track pair_id for result storage

        for pair_id, teams in matchups.items():
            team1_name = teams["team1"]["team_name"]
            team2_name = teams["team2"]["team_name"]

            # Fetch team features for both teams in each pair
            cursor.execute(
                "SELECT * FROM team_features WHERE team_id = %s", (team1_name,)
            )
            team1_features = cursor.fetchone()
            cursor.execute(
                "SELECT * FROM team_features WHERE team_id = %s", (team2_name,)
            )
            team2_features = cursor.fetchone()

            # Concatenate team features for model input
            if team1_features and team2_features:
                combined_features = list(team1_features) + list(team2_features)
                feature_rows.append(combined_features)
                pair_ids.append(pair_id)  # Track pair IDs for later storage

        # Step 2: Call Vertex AI endpoint with batch of combined features
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

        predictions = endpoint.predict(instances=feature_rows).predictions

        # Step 3: Store predictions in MySQL `results` table
        results_data = [
            (pair_id, teams["team1"]["team_name"], teams["team2"]["team_name"], pred)
            for pair_id, teams, pred in zip(pair_ids, matchups.values(), predictions)
        ]

        insert_query = """
            INSERT INTO results (pair_id, team1_name, team2_name, prediction)
            VALUES (%s, %s, %s, %s)
        """

        cursor.executemany(insert_query, results_data)
        conn.commit()  # Commit transaction

        # Close database connection
        cursor.close()
        conn.close()

        # Return a success response
        return jsonify({"status": "Predictions stored successfully"}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "Error", "details": str(e)}), 500

    finally:
        # Clean up connector to avoid open connections
        connector.close()
