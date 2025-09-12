import pandas as pd
import numpy as np


# =====================================
# preprocessing
# =====================================

print("preprocessing")

all_matches = pd.DataFrame()

for year in range (2003, 2025):
    current_file = "./data/atp_matches_" + str(year)+ ".csv"

    year_data = pd.read_csv(current_file)

    all_matches = pd.concat([all_matches, year_data], axis = 0)

# Appropriate to drop so you don’t introduce fake data into features 
# where imputation wouldn’t be meaningful, difficult to estimate ranking and number of winners
# In future we could look at history of recent player matches to get an estimate ranking    

print(all_matches.tail())
all_matches_filtered = all_matches.dropna(subset=[    
    "winner_id","loser_id","winner_ht","loser_ht","winner_age","loser_age",
    "w_ace","w_df","w_svpt","w_1stIn","w_1stWon","w_2ndWon","w_SvGms","w_bpSaved","w_bpFaced",
    "l_ace","l_df","l_svpt","l_1stIn","l_1stWon","l_2ndWon","l_SvGms","l_bpSaved","l_bpFaced",
    "winner_rank_points","loser_rank_points","winner_rank","loser_rank","surface"    
]) 

all_matches_filtered = all_matches_filtered.reset_index(drop=True)

print(all_matches_filtered.tail())

