import json
import psycopg2
from google.cloud import aiplatform
from google.cloud.sql.connector import Connector
from google.auth import default

# Database connection info (replace with actual values)
INSTANCE_CONNECTION_NAME = "YOUR_INSTANCE_CONNECTION_NAME"
DB_USER = "YOUR_DB_USER"
DB_PASS = "YOUR_DB_PASSWORD"
DB_NAME = "YOUR_DB_NAME"

# Vertex AI Model Info
PROJECT_ID = "YOUR_PROJECT_ID"
REGION = "YOUR_REGION"
ENDPOINT_ID = "YOUR_VERTEX_AI_ENDPOINT_ID"

# BigQuery table name for storing predictions
PREDICTION_TABLE = "YOUR_PROJECT.YOUR_DATASET.prediction_table"

# Initialize database connector
connector = Connector()


def get_db_connection():
    return connector.connect(
        INSTANCE_CONNECTION_NAME,
        "pg8000",
        user=DB_USER,
        password=DB_PASS,
        db=DB_NAME,
    )


def predict_matchup(data, context):
    try:
        # Step 1: Parse incoming JSON data
        matchups = json.loads(data["matchups"])

        # Step 2: Connect to Cloud SQL to fetch team features
        conn = get_db_connection()
        cursor = conn.cursor()

        feature_rows = []
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

            # Concatenate team features for the model input
            if team1_features and team2_features:
                combined_features = list(team1_features) + list(team2_features)
                feature_rows.append(combined_features)

        # Step 3: Call Vertex AI endpoint with batch of combined features
        aiplatform.init(project=PROJECT_ID, location=REGION)
        endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)

        predictions = endpoint.predict(instances=feature_rows).predictions

        # Step 4: Store predictions in BigQuery
        from google.cloud import bigquery

        client = bigquery.Client()
        rows_to_insert = [
            {
                "pair_id": pair_id,
                "team1_name": teams["team1"]["team_name"],
                "team2_name": teams["team2"]["team_name"],
                "prediction": pred,
            }
            for pair_id, teams, pred in zip(
                matchups.keys(), matchups.values(), predictions
            )
        ]

        errors = client.insert_rows_json(PREDICTION_TABLE, rows_to_insert)
        if errors:
            print("Errors inserting into BigQuery:", errors)

        # Close database connection
        cursor.close()
        conn.close()

        return {"status": "Predictions stored successfully"}

    except Exception as e:
        print("Error:", e)
        return {"status": "Error", "details": str(e)}

    finally:
        # Clean up connector to avoid open connections
        connector.close()
