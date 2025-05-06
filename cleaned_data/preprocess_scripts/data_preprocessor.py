# import pandas as pd

# # # Load the datasets
# # ball_by_ball_df = pd.read_csv('cleaned_data/data_set_standardized.csv')
# # match_df = pd.read_csv('cleaned_data/match_standardized.csv')

# # # Rename 'id' in match_df to 'match_id' for merging
# # match_df = match_df.rename(columns={'id': 'match_id'})

# # # Select relevant columns from match_df: season, toss_winner, toss_decision, team1, team2
# # match_subset = match_df[['match_id', 'season', 'toss_winner', 'toss_decision', 'team1', 'team2']]

# # # Merge the datasets on match_id
# # merged_df = ball_by_ball_df.merge(match_subset, on='match_id', how='left')

# # # Compute who_is_batting_first
# # # If toss_decision is 'bat', who_is_batting_first is toss_winner
# # # If toss_decision is 'field', who_is_batting_first is the other team
# # merged_df['who_is_batting_first'] = merged_df.apply(
# #     lambda row: row['toss_winner'] if row['toss_decision'] == 'bat'
# #     else (row['team1'] if row['toss_winner'] == row['team2'] else row['team2']),
# #     axis=1
# # )

# # # Drop temporary columns used for computation (team1, team2, toss_winner, toss_decision)
# # merged_df = merged_df.drop(columns=['team1', 'team2', 'player_dismissed',
# #        'dismissal_kind', 'fielder', 'batter', 'bowler', 'non_striker', 'batsman_runs', 'extra_runs', 'extras_type', 'toss_winner', 'toss_decision'])

# # # Save the modified dataset
# # merged_df.to_csv('cleaned_data/modified_data_set.csv', index=False)

# # # Display the first few rows to verify
# # print(merged_df.head())


# # df = pd.read_csv('cleaned_data/modified_data_set.csv')
# # print(df.columns)
# # print(df.head())



# # import pandas as pd

# # Load the modified dataset
# df = pd.read_csv('cleaned_data/modified_data_set.csv')

# # Group by match_id, inning, batting_team, bowling_team, over, season, and who_is_batting_first
# # Aggregate total_runs (sum) and is_wicket (sum for wickets per over)
# over_by_over_df = df.groupby(
#     ['match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'season', 'who_is_batting_first'],
#     as_index=False
# ).agg({
#     'total_runs': 'sum',  # Sum of runs in the over
#     'is_wicket': 'sum'    # Count of wickets in the over
# })

# # Sort the dataframe for better readability
# over_by_over_df = over_by_over_df.sort_values(['match_id', 'inning', 'over'])

# # Save the over-by-over dataset
# over_by_over_df.to_csv('over_by_over_data_set.csv', index=False)

# # Display the first few rows to verify
# print(over_by_over_df.head())





import pandas as pd

# Load the over-by-over dataset
df = pd.read_csv('cleaned_data/over_by_over_data_set.csv')

# Step 1: Make over 1-indexed by adding 1
df['over'] = df['over'] + 1

# Step 2: Compute cumulative runs per match and inning to calculate run_rate
df['cumulative_runs'] = df.groupby(['match_id', 'inning'])['total_runs'].cumsum()

# Calculate run_rate: cumulative runs / number of overs (over is now 1-indexed)
df['run_rate'] = df['cumulative_runs'] / df['over']

# Step 3: Compute target runs for second inning
# First, get total runs for inning 1 for each match
inning1_runs = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
inning1_runs.rename(columns={'total_runs': 'target_runs'}, inplace=True)

# Merge target runs into the main dataframe
df = df.merge(inning1_runs, on='match_id', how='left')

# For inning 1, target_runs is NaN (not applicable); for inning 2, it’s the total from inning 1
# Add 1 to target_runs for inning 2 (since target is the score to beat)
df.loc[df['inning'] == 2, 'target_runs'] = df['target_runs'] + 1
df.loc[df['inning'] == 1, 'target_runs'] = 0  # Not applicable for inning 1

# Step 4: Compute target_left (only for inning 2)
df['target_left'] = 0  # Default for inning 1
df.loc[df['inning'] == 2, 'target_left'] = df['target_runs'] - df['cumulative_runs']
df.loc[df['target_left'] < 0, 'target_left'] = 0  # If target is achieved, set to 0

# Step 5: Compute req_runrate (only for inning 2)
# Assume 20 overs per inning for T20; remaining overs = 20 - current over
df['req_runrate'] = 0  # Default for inning 1
df.loc[df['inning'] == 2, 'req_runrate'] = df['target_left'] / (20 - df['over'] + 1)
df.loc[df['inning'] == 2, 'req_runrate'] = df['req_runrate'].clip(lower=0)  # No negative run rates
df.loc[df['over'] == 20, 'req_runrate'] = 0  # Last over, no required run rate

# Drop temporary columns
df = df.drop(columns=['cumulative_runs'])

# Sort the dataframe for readability
df = df.sort_values(['match_id', 'inning', 'over'])

# Save the updated dataset
df.to_csv('updated_over_by_over_data_set.csv', index=False)

# Display the first few rows to verify
print(df.head())



