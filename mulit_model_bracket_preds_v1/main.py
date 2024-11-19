import json
import yaml
import pymysql
import functions_framework
from google.cloud.sql.connector import Connector
from flask import jsonify
import pickle
import pandas as pd
from google.cloud import storage
import os
import numpy as np

with open("env.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Extract database and Vertex AI configurations
INSTANCE_CONNECTION_NAME = config["INSTANCE_CONNECTION_NAME"]
DB_USER = config["DB_USER"]
DB_PASS = config["DB_PASS"]
DB_NAME = config["DB_NAME"]


def get_db_connection(connector):
    try:

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


def load_from_gcs(file_path, bucket_name="general_bucket_predictive-playmakers"):
    file_name = file_path.split("/")[-1]
    local_path = f"/tmp/{file_name}"

    # Check if the file already exists locally
    if os.path.exists(local_path):
        print(f"File already exists locally: {local_path}")
    else:
        print(f"File not found locally. Downloading from GCS: {file_path}")

        # Initialize the GCS client
        client = storage.Client()

        # Access the bucket and file
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_path)

        # Download the file to local disk
        blob.download_to_filename(local_path)

    # Determine the file type and load the content
    file_type = file_name.split(".")[-1]
    if file_type == "pkl":
        with open(local_path, "rb") as f:
            result = pickle.load(f)
    elif file_type == "csv":
        result = pd.read_csv(local_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    print(f"Retrieved {local_path}")
    return result


def get_team_id(teamA_name, teamB_name):
    teams = load_from_gcs("data/Teams/V1/MTeams.csv")

    team_a = teams.loc[teams["TeamName"] == teamA_name, "TeamID"]
    team_b = teams.loc[teams["TeamName"] == teamB_name, "TeamID"]

    return team_a.values[0], team_b.values[0]


def who_won(teamA_name, seedA, teamB_name, seedB, s_hist_agg, lr, rf, svm):
    print(f"{teamA_name} vs {teamB_name}")
    teamA_id, teamB_id = get_team_id(teamA_name, teamB_name)

    # Code for the input of 2 team Ids
    # team_stats_df = s_hist_agg.set_index('TeamID')
    team_stats_df = s_hist_agg.copy()
    # Trying to build a weighting, but instead used the more recent seasons
    team_stats_df["weight"] = team_stats_df["Season"].apply(
        lambda x: 1 / (2024 - x + 1)
    )
    team_stats_df = team_stats_df.groupby("TeamID", as_index=False).mean()
    team_stats_df = team_stats_df.set_index("TeamID")
    # cols = [col for col in team_stats_df.columns if col not in ['weight', 'Season']]
    # weighted_team_stats = team_stats_df.apply(lambda x: (x[cols] * x['weight']).sum() / x['weight'].sum(),
    #                                           axis=1).reset_index()
    # print(weighted_team_stats)

    # team_stats_df.to_csv("data/sample_stats_1.csv")

    # print(team_stats_df)
    # Grab from f_df
    teamA_stats = team_stats_df.loc[[teamA_id]].add_prefix("A_")
    # print("Team stats A", teamA_stats)
    teamA_stats = teamA_stats[
        [
            "A_Matches_Played",
            "A_Score",
            "A_OppScore",
            "A_FGR",
            "A_OppFGR",
            "A_ORCHANCE",
            "A_OppORCHANCE",
            "A_ORR",
            "A_OppORR",
            "A_PFDIFF",
            "A_OppPFDIFF",
            "A_FGADIFF",
            "A_OppFGADIFF",
            "A_FGA2",
            "A_OppFGA2",
            "A_FG2PCT",
            "A_OppFG2PCT",
            "A_POSS",
            "A_OppPOSS",
            "A_PPP",
            "A_OppPPP",
            "A_OER",
            "A_OppOER",
            "A_DER",
            "A_OppDER",
            "A_NET_RAT",
            "A_OppNET_RAT",
            "A_DRR",
            "A_OppDRR",
            "A_AstRatio",
            "A_OppAstRatio",
            "A_Pace",
            "A_OppPace",
            "A_FTARate",
            "A_OppFTARate",
            "A_3PAR",
            "A_Opp3PAR",
            "A_eFG",
            "A_OppeFG",
            "A_TRR",
            "A_OppTRR",
            "A_Astper",
            "A_OppAstper",
            "A_Stlper",
            "A_OppStlper",
            "A_Blkper",
            "A_OppBlkper",
            "A_PPSA",
            "A_OppPPSA",
            "A_FGperc",
            "A_OppFGperc",
            "A_3Fper",
            "A_Opp3Fper",
            "A_ToRatio",
            "A_OppToRatio",
            "A_Matches_Won",
            "A_WinRatio",
            "A_LossRatio",
            "A_AvgScore",
            "A_AvgOppScore",
            "A_ScoreDiff",
            "A_TODIFF",
            "A_STLDIFF",
            "A_BLKDIFF",
        ]
    ]
    # teamA_stats = teamA_stats['A_Matches_Played', 'A_Score',
    # 'A_OppScore', 'A_NumOT']
    teamB_stats = team_stats_df.loc[[teamB_id]].add_prefix("B_")
    teamB_stats = teamB_stats[
        [
            "B_Matches_Played",
            "B_Score",
            "B_OppScore",
            "B_FGR",
            "B_OppFGR",
            "B_ORCHANCE",
            "B_OppORCHANCE",
            "B_ORR",
            "B_OppORR",
            "B_PFDIFF",
            "B_OppPFDIFF",
            "B_FGADIFF",
            "B_OppFGADIFF",
            "B_FGA2",
            "B_OppFGA2",
            "B_FG2PCT",
            "B_OppFG2PCT",
            "B_POSS",
            "B_OppPOSS",
            "B_PPP",
            "B_OppPPP",
            "B_OER",
            "B_OppOER",
            "B_DER",
            "B_OppDER",
            "B_NET_RAT",
            "B_OppNET_RAT",
            "B_DRR",
            "B_OppDRR",
            "B_AstRatio",
            "B_OppAstRatio",
            "B_Pace",
            "B_OppPace",
            "B_FTARate",
            "B_OppFTARate",
            "B_3PAR",
            "B_Opp3PAR",
            "B_eFG",
            "B_OppeFG",
            "B_TRR",
            "B_OppTRR",
            "B_Astper",
            "B_OppAstper",
            "B_Stlper",
            "B_OppStlper",
            "B_Blkper",
            "B_OppBlkper",
            "B_PPSA",
            "B_OppPPSA",
            "B_FGperc",
            "B_OppFGperc",
            "B_3Fper",
            "B_Opp3Fper",
            "B_ToRatio",
            "B_OppToRatio",
            "B_Matches_Won",
            "B_WinRatio",
            "B_LossRatio",
            "B_AvgScore",
            "B_AvgOppScore",
            "B_ScoreDiff",
            "B_TODIFF",
            "B_STLDIFF",
            "B_BLKDIFF",
        ]
    ]

    # new_game_features = pd.DataFrame([{**teamA_stats, **teamB_stats}])
    new_game_features = pd.concat(
        [teamA_stats.reset_index(drop=True), teamB_stats.reset_index(drop=True)],
        axis=1,
        ignore_index=True,
    )

    # print("The new game features", new_game_features)

    # print(teamB_stats)

    lr_prob = lr.predict_proba(new_game_features)
    rf_prob = rf.predict_proba(new_game_features)
    svm_prob = svm.predict_proba(new_game_features)

    print(
        f"The Win probability for the LR {lr_prob}, the Random Forest {rf_prob} and the SVM {svm_prob}"
    )

    lr_team, rf_team, svm_team = 0, 0, 0
    lr_p, rf_p, sv_p = 0, 0, 0

    if lr_prob[0][1] > lr_prob[0][0]:
        lr_team_w = teamB_id
        lr_team_l = teamA_id
        lr_p = lr_prob[0][1]
    else:
        lr_team_w = teamA_id
        lr_team_l = teamB_id
        lr_p = lr_prob[0][0]

    if rf_prob[0][1] > rf_prob[0][0]:
        rf_team = teamB_id
        rf_p = rf_prob[0][1]
    else:
        rf_team = teamA_id
        rf_p = rf_prob[0][0]

    if svm_prob[0][1] > svm_prob[0][0]:
        svm_team = teamB_id
        svm_p = svm_prob[0][1]
    else:
        svm_team = teamA_id
        svm_p = svm_prob[0][0]

    # Some Cinderella Weighting Calculation

    # Example of how to pick the winner and loser above

    winner = ""
    if teamB_id == lr_team_w:
        winner = teamB_name
    else:
        winner = teamA_name

    new_dict = {}
    new_dict["TeamA"] = teamA_name
    new_dict["TeamA_Avg Points Per Game: "] = teamA_stats["A_AvgScore"].iloc[0]
    new_dict["TeamA_Turnover Ratio: "] = teamA_stats["A_TODIFF"].iloc[0]
    new_dict["TeamA_Efficiency Rating: "] = teamA_stats["A_NET_RAT"].iloc[0]
    new_dict["TeamB"] = teamB_name
    new_dict["TeamB_Avg Points Per Game: "] = teamB_stats["B_AvgScore"].iloc[0]
    new_dict["TeamB_Turnover Ratio: "] = teamB_stats["B_TODIFF"].iloc[0]
    new_dict["TeamB_Efficiency Rating: "] = teamB_stats["B_NET_RAT"].iloc[0]
    new_dict["Winner"] = winner
    print(f"winner: {winner}")

    return lr_prob, rf_prob, svm_prob, new_dict


@functions_framework.http
def predict_bracket(request):
    """HTTP Cloud Function to process iris sample IDs, call Vertex AI, and store results."""
    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "starting_bracket" not in request_json:
        return "Invalid input: 'starting_bracket' list required", 400

    starting_bracket = request_json["starting_bracket"]

    print("Starting Bracket:", starting_bracket)
    try:
        # step 1: load the features to use for the results
        features = load_from_gcs("data/prediction_features/V1/team_features.csv")

        # Step 2: get predictions and explanations
        # step 2.1 download the models from storage and load them
        lr = load_from_gcs("models/main_models/test_V0.1/lr_model_v1.0.pkl")

        rf = load_from_gcs(
            "models/main_models/test_V0.1/rf_model_v1.0.pkl",
        )

        svm = load_from_gcs(
            "models/main_models/test_V0.1/svm_model_v1.0.pkl",
        )

        # step 2.2 get preds
        total_teams = len(starting_bracket) * 2
        rounds = int(np.log2(total_teams))

        print(f"Total Teams: {total_teams}, Total Rounds: {rounds}")

        current_round = starting_bracket
        results = {}  # Dictionary to store the results

        for round_idx in range(rounds):  # Iterate through rounds
            print(f"\nProcessing Round {round_idx}")
            next_round = []

            for match_idx, matchup in enumerate(current_round):
                lr_prob, rf_prob, svm_prob, game_stats = who_won(
                    teamA_name=matchup["teamA"],
                    seedA=matchup["seedA"],
                    teamB_name=matchup["teamB"],
                    seedB=matchup["seedB"],
                    s_hist_agg=features,
                    lr=lr,
                    rf=rf,
                    svm=svm,
                )
                results[f"{round_idx}-{match_idx}"] = game_stats

                # Collect winners for the next round
                next_round.append(game_stats["Winner"])

            if len(next_round) != 1:
                # Create matches for the next round
                current_round = [
                    {
                        "teamA": next_round[j],
                        "seedA": j,
                        "teamB": next_round[j + 1],
                        "seedB": j + 1,
                    }
                    for j in range(0, len(next_round), 2)
                ]

        # step 2.3 get explanations
        # predictions = results

        # get_shap_tree_explainer = pkl.loads(model_path)
        # shap_preds = get_shap_tree_explainer(model, feature_rows)

        # print("Received predictions from Vertex AI")
        print("Predictions:", results)
        return jsonify(results), 200

        # return jsonify({"predictions": results_data}), 200

        # Step 3v2: Store predictions in MySQL `iris_results` table
        # results_data = [(iris_id, pred) for iris_id, pred in zip(iris_ids, predictions)]

        # insert_query = """
        #     INSERT INTO iris_results (id, prediction, shap_values)
        #     VALUES (%s, %s)
        # """

        # cursor.executemany(insert_query, results_data)
        # conn.commit()  # Commit transaction
        # print("Stored predictions in Cloud SQL")

        # # Close database connection
        # cursor.close()
        # conn.close()

        # Return a success response
        # return jsonify({"status": "Predictions stored successfully"}), 200
        # return jsonify({"predictions": results_data}), 200

    except Exception as e:
        print("Error:", e)
        return jsonify({"status": "Error", "details": str(e)}), 500

    finally:
        pass
        # Clean up connector to avoid open connections
        # connector.close()
