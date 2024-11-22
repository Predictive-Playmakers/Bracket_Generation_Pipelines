import yaml
import functions_framework
from flask import jsonify, make_response
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


def load_from_gcs(file_path, bucket_name="general_bucket_predictive-playmakers"):
    file_name = file_path.split("/")[-1]
    local_path = f"/tmp/{file_name}"

    # Check if the file already exists locally
    if os.path.exists(local_path):
        # print(f"File already exists locally: {local_path}")
        pass
    else:
        # print(f"File not found locally. Downloading from GCS: {file_path}")

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


def cinderella_weight(
    teamA_stats, teamB_stats, seedA, seedB, teamA_id, teamB_id, kpom, rpi
):

    a_rpi = rpi[["ATeamID", "A_RPI"]].groupby("ATeamID")["A_RPI"].mean().reset_index()
    b_rpi = rpi[["BTeamID", "B_RPI"]].groupby("BTeamID")["B_RPI"].mean().reset_index()
    # print("The rpi's", a_rpi, b_rpi)

    weight = 1.0

    if (seedB - seedA) >= 5:
        weight += 1.50
    if "B_AvgScore" in teamB_stats and "A_AvgScore" in teamA_stats:
        if np.int32(teamB_stats["B_AvgScore"]) > np.int32(teamA_stats["A_AvgScore"]):
            weight += 0.55
    if teamA_id in kpom["TeamID"].values and teamB_id in kpom["TeamID"].values:
        if (
            kpom.loc[kpom["TeamID"] == teamA_id, "luck"].values[0]
            > kpom.loc[kpom["TeamID"] == teamB_id, "luck"].values[0]
        ):
            weight += 0.25
    if teamA_id in a_rpi["ATeamID"].values and teamB_id in b_rpi["BTeamID"].values:
        if (
            b_rpi.loc[b_rpi["BTeamID"] == teamA_id, "B_RPI"].values[0]
            > a_rpi.loc[a_rpi["ATeamID"] == teamB_id, "A_RPI"].values[0]
        ):
            weight += -0.15

    # print("cinderella weight")
    return weight


def shap_to_json(explainer, winner_id, X_sample):
    """
    Convert SHAP output to JSON compatible with the React visualization package.

    Parameters:
    - explainer: SHAP explainer object
    - X_sample: Input data (single sample or multiple rows as a DataFrame)

    Returns:
    - JSON object as a Python dictionary
    """
    shap_values = explainer.shap_values(X_sample.iloc[0])
    # print(shap_values.shape)

    # Base value (expected value of the model output)
    base_value = explainer.expected_value[0]  # Assuming a single output model
    shap_json = {"data": []}

    for i in range(X_sample.shape[0]):  # Iterate over each sample
        shap_data = {
            "outNames": ["output value"],
            "baseValue": base_value,
            "outValue": base_value
            + sum(shap_value[winner_id] for shap_value in shap_values),
            "link": "identity",
            "featureNames": X_sample.columns.tolist(),
            "features": {},
            "plot_cmap": "DrDb",
            "labelMargin": 20,
        }

        # Add feature details
        for j, feature_name in enumerate(X_sample.columns):
            shap_data["features"][str(j)] = {
                "effect": shap_values[j][winner_id],
                "value": X_sample.iloc[i, j],
            }

        shap_json["data"].append(shap_data)

    return shap_json


def who_won(
    teamA_name, seedA, teamB_name, seedB, s_hist_agg, lr, rf, kpom, rpi, rf_explainer
):

    teamA_id, teamB_id = get_team_id(teamA_name, teamB_name)

    # Code for the input of 2 team Ids
    team_stats_df = s_hist_agg.copy()

    team_stats_df = team_stats_df.set_index("TeamID")
    # Grab from f_df
    teamA_stats = team_stats_df.loc[[teamA_id]].add_prefix("A_")
    teamA_stats_adj = teamA_stats[
        [
            "A_Score",
            "A_OppScore",
            "A_NumOT",
            "A_FGM",
            "A_FGA",
            "A_FGM3",
            "A_FGA3",
            "A_FTM",
            "A_FTA",
            "A_OR",
            "A_DR",
            "A_Ast",
            "A_TO",
            "A_Stl",
            "A_Blk",
            "A_PF",
            "A_OppNET_RAT",
        ]
    ]

    teamB_stats = team_stats_df.loc[[teamB_id]].add_prefix("B_")
    teamB_stats_adj = teamB_stats[
        [
            "B_Score",
            "B_OppScore",
            "B_NumOT",
            "B_FGM",
            "B_FGA",
            "B_FGM3",
            "B_FGA3",
            "B_FTM",
            "B_FTA",
            "B_OR",
            "B_DR",
            "B_Ast",
            "B_TO",
            "B_Stl",
            "B_Blk",
            "B_PF",
            "B_OppNET_RAT",
        ]
    ]

    new_game_features = pd.concat(
        [
            teamA_stats_adj.reset_index(drop=True),
            teamB_stats_adj.reset_index(drop=True),
        ],
        axis=1,
    )

    lr_prob = lr.predict_proba(new_game_features)
    rf_prob = rf.predict_proba(new_game_features)
    # print(f"The Win probability for the LR {lr_prob}, the Random Forest {rf_prob}")

    # Cinderella Weighting Calculation

    c_weight = cinderella_weight(
        teamA_stats, teamB_stats, seedA, seedB, teamA_id, teamB_id, kpom, rpi
    )
    # print("The cinderella weight: ", c_weight)

    win_prob = ((np.int32(lr_prob[0][1] * c_weight) + np.int32(rf_prob[0][1])) / 2) - (
        ((np.int32(lr_prob[0][0]) * c_weight) + np.int32(rf_prob[0][0])) / 2
    )

    y = 0
    # print("The Win Probability", win_prob)
    if win_prob > 0:
        winner = teamA_name
        y = 1
    else:
        winner = teamB_name

    # Output Dictionary of the winnner
    winner_dict = {}
    winner_dict["TeamA"] = teamA_name
    winner_dict["TeamA_Avg Points Per Game: "] = teamA_stats["A_AvgScore"].iloc[0]
    winner_dict["TeamA_Turnover Ratio: "] = teamA_stats["A_TODIFF"].iloc[0]
    winner_dict["TeamA_Efficiency Rating: "] = teamA_stats["A_NET_RAT"].iloc[0]
    winner_dict["TeamB"] = teamB_name
    winner_dict["TeamB_Avg Points Per Game: "] = teamB_stats["B_AvgScore"].iloc[0]
    winner_dict["TeamB_Turnover Ratio: "] = teamB_stats["B_TODIFF"].iloc[0]
    winner_dict["TeamB_Efficiency Rating: "] = teamB_stats["B_NET_RAT"].iloc[0]
    winner_dict["Winner"] = winner

    shap_json = shap_to_json(rf_explainer, y, new_game_features)

    return winner_dict, shap_json


@functions_framework.http
def predict_bracket(request):
    """HTTP Cloud Function to process iris sample IDs, call Vertex AI, and store results."""
    # handle CORS Policy
    if request.method == "OPTIONS":
        response = make_response("", 204)  # No content for preflight
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    # Parse JSON input from request
    request_json = request.get_json(silent=True)
    if not request_json or "starting_bracket" not in request_json:
        response = make_response(
            "Invalid input: 'starting_bracket' key with a list value pair required", 400
        )
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    starting_bracket = request_json["starting_bracket"]

    # print("Starting Bracket:", starting_bracket)
    try:
        # Load features and models
        features = load_from_gcs("data/prediction_features/V2/team_features.csv")
        kpom = load_from_gcs("data/prediction_features/V2/kenpom_luck_teamID.csv")
        rpi = load_from_gcs("data/prediction_features/V2/rpi_data.csv")

        lr = load_from_gcs("models/main_models/V2.0/lr_model_v2.0.pkl")
        rf = load_from_gcs("models/main_models/V2.0/rf_model_v2.0.pkl")
        rf_explainer = load_from_gcs("models/main_models/V2.0/rf_explainerV2.0.pkl")

        # soon add in the explainer!

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
                    game_stats, shap_json = who_won(
                        teamA_name=teamA["name"],
                        seedA=teamA["seed"],
                        teamB_name=teamB["name"],
                        seedB=teamB["seed"],
                        s_hist_agg=features,
                        lr=lr,
                        rf=rf,
                        kpom=kpom,
                        rpi=rpi,
                        rf_explainer=rf_explainer,
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
                            "shap": shap_json,
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
        sf_game_stats_1, shap_json = who_won(
            teamA_name=division_winners["division0"]["name"],
            seedA=division_winners["division0"]["seed"],
            teamB_name=division_winners["division1"]["name"],
            seedB=division_winners["division1"]["seed"],
            s_hist_agg=features,
            lr=lr,
            rf=rf,
            kpom=kpom,
            rpi=rpi,
            rf_explainer=rf_explainer,
        )

        finals["semifinals"].append(
            {
                "id": "sf-1",
                "teams": [division_winners["division0"], division_winners["division1"]],
                "results": sf_game_stats_1,
                "shap": shap_json,
            }
        )

        sf_game_stats_2, shap_json = who_won(
            teamA_name=division_winners["division2"]["name"],
            seedA=division_winners["division2"]["seed"],
            teamB_name=division_winners["division3"]["name"],
            seedB=division_winners["division3"]["seed"],
            s_hist_agg=features,
            lr=lr,
            rf=rf,
            kpom=kpom,
            rpi=rpi,
            rf_explainer=rf_explainer,
        )
        finals["semifinals"].append(
            {
                "id": "sf-2",
                "teams": [division_winners["division2"], division_winners["division3"]],
                "results": sf_game_stats_2,
                "shap": shap_json,
            }
        )

        # Process championship
        final_game_stats, shap_json = who_won(
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
            kpom=kpom,
            rpi=rpi,
            rf_explainer=rf_explainer,
        )

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
            "shap": shap_json,
        }

        # Combine results and finals
        results["finals"] = finals

        results_dict = {"results": results}

        # Build response with CORS headers
        response = make_response(jsonify(results_dict), 200)
        response.headers["Access-Control-Allow-Origin"] = "*"  # Allow all origins
        return response

    except Exception as e:
        print("Error:", e)
        response = make_response(jsonify({"status": "Error", "details": str(e)}), 500)
        response.headers["Access-Control-Allow-Origin"] = "*"  # Add CORS headers
        return response
