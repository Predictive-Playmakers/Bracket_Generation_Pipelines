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
        # print(f"File already exists locally: {local_path}")
        pass
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

    # print(f"Retrieved {local_path}")
    return result


def get_team_id(teamA_name, teamB_name):
    teams = load_from_gcs("data/Teams/V1/MTeams.csv")

    team_a = teams.loc[teams["TeamName"] == teamA_name, "TeamID"]
    team_b = teams.loc[teams["TeamName"] == teamB_name, "TeamID"]

    return team_a.values[0], team_b.values[0]


def who_won(teamA_name, seedA, teamB_name, seedB, s_hist_agg, lr, rf, svm):
    # print(f"{teamA_name} vs {teamB_name}")
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

    # print(
    #     f"The Win probability for the LR {lr_prob}, the Random Forest {rf_prob} and the SVM {svm_prob}"
    # )

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
    # print(f"winner: {winner}")

    return lr_prob, rf_prob, svm_prob, new_dict


@functions_framework.http
def predict_bracket(request):
    """HTTP Cloud Function to process iris sample IDs, call Vertex AI, and store results."""
    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "starting_bracket" not in request_json:
        return (
            "Invalid input: 'starting_bracket' key with a list value pair required",
            400,
        )

    starting_bracket = request_json["starting_bracket"]

    # print("Starting Bracket:", starting_bracket)
    try:
        # Load features and models
        features = load_from_gcs("data/prediction_features/V1/team_features.csv")
        lr = load_from_gcs("models/main_models/test_V0.1/lr_model_v1.0.pkl")
        rf = load_from_gcs("models/main_models/test_V0.1/rf_model_v1.0.pkl")
        svm = load_from_gcs("models/main_models/test_V0.1/svm_model_v1.0.pkl")

        results = {}  # Dictionary to store division results
        finals = {"semifinals": [], "championship": {}}  # Structure for finals

        # Process each division
        division_winners = {}
        for division in starting_bracket.keys():
            if "division" not in division:
                # Skip non-division keys
                continue

            # Initialize results for this division
            division_results = {f"round0": []}
            current_round = starting_bracket[division]["round0"]
            round_idx = 0

            while (
                current_round
            ):  # Continue processing while there are matches to process
                next_round = []
                division_results[f"round{round_idx}"] = []

                for match_idx, match in enumerate(current_round):
                    teamA = match["teams"][0]
                    teamB = match["teams"][1]

                    # Call the prediction function
                    lr_prob, rf_prob, svm_prob, game_stats = who_won(
                        teamA_name=teamA["name"],
                        seedA=teamA["seed"],
                        teamB_name=teamB["name"],
                        seedB=teamB["seed"],
                        s_hist_agg=features,
                        lr=lr,
                        rf=rf,
                        svm=svm,
                    )

                    # Add game stats to results for the current round and division
                    match_id = f"{round_idx}-{match_idx}"
                    division_results[f"round{round_idx}"].append(
                        {
                            "id": match_id,
                            "teams": [
                                {
                                    "name": teamA["name"],
                                    "seed": teamA["seed"],
                                    "score": teamA.get("score", 0),
                                },
                                {
                                    "name": teamB["name"],
                                    "seed": teamB["seed"],
                                    "score": teamB.get("score", 0),
                                },
                            ],
                            "result": game_stats,
                        }
                    )

                    # Determine the winner and prepare for the next round
                    winner = game_stats["Winner"]
                    next_round.append(
                        {
                            "id": f"{round_idx + 1}-{len(next_round)}",
                            "teams": [
                                {
                                    "name": winner,
                                    "seed": (
                                        teamA["seed"]
                                        if winner == teamA["name"]
                                        else teamB["seed"]
                                    ),
                                }
                            ],
                        }
                    )

                # Check if this is the final round
                if len(next_round) == 1:
                    winner_team = next_round[0]["teams"][0]
                    division_winners[division] = winner_team
                    break  # Exit the loop since the final winner is determined

                # Update current round and increment round index
                current_round = [
                    {
                        "id": f"{round_idx + 1}-{idx // 2}",
                        "teams": [
                            next_round[idx]["teams"][0],
                            next_round[idx + 1]["teams"][0],
                        ],
                    }
                    for idx in range(0, len(next_round), 2)
                ]
                round_idx += 1

            # Store the division results
            results[division] = division_results

        # Process semifinals
        sf_game_stats_1 = who_won(
            teamA_name=division_winners["division0"]["name"],
            seedA=division_winners["division0"]["seed"],
            teamB_name=division_winners["division1"]["name"],
            seedB=division_winners["division1"]["seed"],
            s_hist_agg=features,
            lr=lr,
            rf=rf,
            svm=svm,
        )[
            3
        ]  # Extract game stats

        finals["semifinals"].append(
            {
                "id": "sf-1",
                "teams": [division_winners["division0"], division_winners["division1"]],
                "results": sf_game_stats_1,
            }
        )

        sf_game_stats_2 = who_won(
            teamA_name=division_winners["division2"]["name"],
            seedA=division_winners["division2"]["seed"],
            teamB_name=division_winners["division3"]["name"],
            seedB=division_winners["division3"]["seed"],
            s_hist_agg=features,
            lr=lr,
            rf=rf,
            svm=svm,
        )[
            3
        ]  # Extract game stats

        finals["semifinals"].append(
            {
                "id": "sf-2",
                "teams": [division_winners["division2"], division_winners["division3"]],
                "results": sf_game_stats_2,
            }
        )

        # Process championship
        final_game_stats = who_won(
            teamA_name=sf_game_stats_1["Winner"],
            seedA=next(
                team["seed"]
                for team in finals["semifinals"][0]["teams"]
                if team["name"] == sf_game_stats_1["Winner"]
            ),
            teamB_name=sf_game_stats_2["Winner"],
            seedB=next(
                team["seed"]
                for team in finals["semifinals"][1]["teams"]
                if team["name"] == sf_game_stats_2["Winner"]
            ),
            s_hist_agg=features,
            lr=lr,
            rf=rf,
            svm=svm,
        )[
            3
        ]  # Extract game stats

        finals["championship"] = {
            "id": "final",
            "teams": [
                {
                    "name": sf_game_stats_1["Winner"],
                    "seed": next(
                        team["seed"]
                        for team in finals["semifinals"][0]["teams"]
                        if team["name"] == sf_game_stats_1["Winner"]
                    ),
                    "score": 0,
                },
                {
                    "name": sf_game_stats_2["Winner"],
                    "seed": next(
                        team["seed"]
                        for team in finals["semifinals"][1]["teams"]
                        if team["name"] == sf_game_stats_2["Winner"]
                    ),
                    "score": 0,
                },
            ],
            "results": final_game_stats,
        }

        # Combine results and finals
        results["finals"] = finals

        return {"results": results}

    except Exception as e:
        print("Error:", e)
        return {"status": "Error", "details": str(e)}

        # take the div0 round3 winner and div1 round 3 winner and insert them into
        # results['finals']['']

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
