# ============================================
# 🎾 1. Import Libraries
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

pd.set_option('display.max_columns', None)
os.makedirs("./images", exist_ok=True)
os.makedirs("./data", exist_ok=True)

# ============================================
# 🎾 2. Load Historical Matches (2000-2024)
# ============================================
print("Loading historical data...")
all_matches = pd.DataFrame()
for year in range(2000, 2025):
    file_path = "../data/atp_matches_" + str(year)+ ".csv"
    year_data = pd.read_csv(file_path)
    all_matches = pd.concat([all_matches, year_data], axis=0)

# Drop rows with critical NaNs
all_matches = all_matches.dropna(subset=[
    "winner_id","loser_id","winner_ht","loser_ht","winner_age","loser_age",
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
    "winner_rank_points","loser_rank_points","winner_rank","loser_rank","surface"
]).reset_index(drop=True)

print(f"Historical matches shape: {all_matches.shape}")

# ============================================
# 🎾 3. Load Australian Open 2025 (Test Set)
# ============================================
aus_open_data = pd.read_csv("../data/aus_open_2025.csv")
print(f"AU Open 2025 matches shape: {aus_open_data.shape}")

# ============================================
# 🎾 4. Feature Engineering Function
# ============================================
def build_features(df, h2h_dict=None, h2h_surface_dict=None, elo_players=None, elo_surfaces=None):
    """
    Build match features: ATP diff, age, height, H2H, Elo.
    If H2H/Elo dicts are provided, they will be updated progressively.
    """
    n = len(df)
    final_df = pd.DataFrame()
    final_df["Winner_ID"] = df["winner_id"]
    final_df["LOSER_ID"] = df["loser_id"]
    final_df["ATP_POINT_DIff"] = df["winner_rank_points"] - df["loser_rank_points"]
    final_df["ATP_RANK_DIff"] = df["winner_rank"] - df["loser_rank"]
    final_df["AGE_DIFF"] = df["winner_age"] - df["loser_age"]
    final_df["HEIGHT_DIFF"] = df["winner_ht"] - df["loser_ht"]
    final_df["BEST_OF"] = df["best_of"]
    final_df["DRAW_SIZE"] = df["draw_size"]

    # Initialize H2H and Elo if not provided
    if h2h_dict is None:
        h2h_dict = defaultdict(int)
    if h2h_surface_dict is None:
        h2h_surface_dict = defaultdict(lambda: defaultdict(int))
    if elo_players is None:
        elo_players = defaultdict(lambda: 1400)
    if elo_surfaces is None:
        elo_surfaces = defaultdict(lambda: defaultdict(lambda: 1400))

    total_h2h = []
    total_h2h_surface = []
    df_elo_diff = []
    df_elo_surfaces_diff = []

    for idx, (winner, loser, surface) in enumerate(tqdm(zip(df['winner_id'], df['loser_id'], df['surface']), total=n)):
        # H2H
        wins = h2h_dict[(winner, loser)]
        loses = h2h_dict[(loser, winner)]
        wins_surface = h2h_surface_dict[surface][(winner, loser)]
        loses_surface = h2h_surface_dict[surface][(loser, winner)]

        total_h2h.append(wins - loses)
        total_h2h_surface.append(wins_surface - loses_surface)

        h2h_dict[(winner, loser)] += 1
        h2h_surface_dict[surface][(winner, loser)] += 1

        # Elo overall
        k = 32
        elo_w = elo_players[winner]
        elo_l = elo_players[loser]
        E_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        E_l = 1 / (1 + 10 ** ((elo_w - elo_l) / 400))
        elo_w += k * (1 - E_w)
        elo_l += k * (0 - E_l)
        elo_players[winner] = elo_w
        elo_players[loser] = elo_l
        df_elo_diff.append(elo_w - elo_l)

        # Elo surface
        elo_w_s = elo_surfaces[surface][winner]
        elo_l_s = elo_surfaces[surface][loser]
        E_w_s = 1 / (1 + 10 ** ((elo_l_s - elo_w_s) / 400))
        E_l_s = 1 / (1 + 10 ** ((elo_w_s - elo_l_s) / 400))
        elo_w_s += k * (1 - E_w_s)
        elo_l_s += k * (0 - E_l_s)
        elo_surfaces[surface][winner] = elo_w_s
        elo_surfaces[surface][loser] = elo_l_s
        df_elo_surfaces_diff.append(elo_w_s - elo_l_s)

    final_df["H2H_DIFF"] = total_h2h
    final_df["H2H_DIFF_SURFACE"] = total_h2h_surface
    final_df["ELO_DIFF"] = df_elo_diff
    final_df["ELO_SURFACE_DIFF"] = df_elo_surfaces_diff

    return final_df, h2h_dict, h2h_surface_dict, elo_players, elo_surfaces

# ============================================
# 🎾 5. Build Training Features
# ============================================
print("Building training features...")
final_data_train, h2h_dict, h2h_surface_dict, elo_players, elo_surfaces = build_features(all_matches)

# ============================================
# 🎾 6. Prepare Data for ML
# ============================================
feature_cols = [
    "ATP_POINT_DIff", "ATP_RANK_DIff", "AGE_DIFF", "HEIGHT_DIFF",
    "BEST_OF", "DRAW_SIZE", "H2H_DIFF", "H2H_DIFF_SURFACE", 
    "ELO_DIFF", "ELO_SURFACE_DIFF"
]

# Winner examples
X_winner = final_data_train[feature_cols].copy()
y_winner = np.ones(len(X_winner))

# Loser examples (flip signs)
X_loser = -X_winner
y_loser = np.zeros(len(X_winner))

# Combine
X_total = pd.concat([X_winner, X_loser], axis=0)
y_total = np.concatenate([y_winner, y_loser])

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_total, y_total, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ============================================
# 🎾 7. Train Models
# ============================================

# --- SVM ---
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_preds = svm_model.predict(X_val_scaled)
svm_acc = accuracy_score(y_val, svm_preds)
print(f"SVM Validation Accuracy: {svm_acc:.4f}")

# --- MLP ---
mlp_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp_model.fit(X_train_scaled, y_train)
mlp_preds = mlp_model.predict(X_val_scaled)
mlp_acc = accuracy_score(y_val, mlp_preds)
print(f"MLP Validation Accuracy: {mlp_acc:.4f}")

# ============================================
# 🎾 8. Build Test Features (AU Open 2025)
# ============================================
print("Building test features for AU Open 2025...")
final_data_test, _, _, _, _ = build_features(
    aus_open_data,
    h2h_dict=h2h_dict,
    h2h_surface_dict=h2h_surface_dict,
    elo_players=elo_players,
    elo_surfaces=elo_surfaces
)

# Drop any missing rows (safety)
final_data_test = final_data_test.dropna()
X_test = final_data_test[feature_cols].copy()
X_test_scaled = scaler.transform(X_test)

# ============================================
# 🎾 9. Predict AU Open 2025 Winners
# ============================================
aus_preds = mlp_model.predict(X_test_scaled)
aus_probs = mlp_model.predict_proba(X_test_scaled)[:, 1]
aus_open_data["Predicted_Winner"] = aus_preds
aus_open_data["Win_Probability"] = aus_probs

# Optional: compute test accuracy if actual outcomes exist
if "actual_label" in aus_open_data.columns:
    test_acc = accuracy_score(aus_open_data["actual_label"], aus_preds)
    print(f"AU Open 2025 Test Accuracy: {test_acc:.4f}")
else:
    print("No actual outcomes available — predictions only.")

# Save predictions
aus_open_data.to_csv("./data/aus_open_2025_predictions.csv", index=False)
print("✅ Predictions saved to './data/aus_open_2025_predictions.csv'")