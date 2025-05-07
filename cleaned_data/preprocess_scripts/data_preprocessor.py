import pandas as pd

def derive_features_from_ball_data(deliveries_df, matches_df):
    """Derive all required features from ball-by-ball data.
    
    Args:
        deliveries_df: DataFrame containing ball-by-ball data
        matches_df: DataFrame containing match metadata
    
    Returns:
        DataFrame with derived features for each over
    """
    # Merge match data with deliveries
    df = pd.merge(deliveries_df, matches_df[['id', 'season', 'date', 'team1', 'team2']], 
                 left_on='match_id', right_on='id')
    
    # Group by match_id, inning, over to get per-over statistics
    over_stats = df.groupby(['match_id', 'inning', 'over']).agg(
        total_runs=('total_runs', 'sum'),
        is_wicket=('is_wicket', 'sum'),
        batting_team=('batting_team', 'first'),
        bowling_team=('bowling_team', 'first'),
        season=('season', 'first')
    ).reset_index()

    # Sort by match_id, inning, over for sequential processing
    over_stats = over_stats.sort_values(['match_id', 'inning', 'over'])
    
    # Calculate cumulative statistics
    result = []
    for (match_id, inning), group in over_stats.groupby(['match_id', 'inning']):
        cum_runs = 0
        cum_wickets = 0
        weighted_rr = 0
        alpha = 0.7  # Weight for weighted run rate calculation
        
        for _, row in group.iterrows():
            # Update cumulative stats
            cum_runs += row['total_runs']
            cum_wickets += row['is_wicket']
            
            # Calculate run rates
            current_run_rate = row['total_runs']  # Runs in current over
            run_rate = cum_runs / row['over'] if row['over'] > 0 else 0
            
            # Calculate weighted run rate
            if row['over'] == 1:
                weighted_rr = current_run_rate
            else:
                weighted_rr = alpha * current_run_rate + (1-alpha) * weighted_rr
            
            # Get target and required run rate for second innings
            target = 0
            target_left = 0
            req_runrate = 0
            
            if inning == 2:
                # Get first innings total
                first_inning = over_stats[(over_stats['match_id'] == match_id) & 
                                        (over_stats['inning'] == 1)]
                if not first_inning.empty:
                    target = first_inning['total_runs'].sum() + 1
                    target_left = target - cum_runs
                    remaining_overs = 20 - row['over']
                    req_runrate = target_left / remaining_overs if remaining_overs > 0 else 99.99
            
            # Store over summary
            result.append({
                'match_id': match_id,
                'inning': inning,
                'over': row['over'],
                'batting_team': row['batting_team'],
                'bowling_team': row['bowling_team'],
                'season': row['season'],
                'total_runs': row['total_runs'],
                'is_wicket': row['is_wicket'],
                'cum_runs': cum_runs,
                'cum_wickets': cum_wickets,
                'run_rate': run_rate,
                'curr_run_rate': current_run_rate,
                'weighted_run_rate': weighted_rr,
                'target': target,
                'target_left': target_left,
                'req_runrate': req_runrate
            })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    
    # Add 'is_powerplay' feature (first 6 overs)
    result_df['is_powerplay'] = result_df['over'] <= 6
    
    # Add 'balls_remaining' feature
    result_df['balls_remaining'] = (20 - result_df['over']) * 6
    
    # Add next over stats for training (shifted by 1 within each match-inning)
    result_df['next_over_runs'] = result_df.groupby(['match_id', 'inning'])['total_runs'].shift(-1)
    result_df['next_over_wickets'] = result_df.groupby(['match_id', 'inning'])['is_wicket'].shift(-1)
    
    # Drop last over of each innings (no next over to predict)
    result_df = result_df.dropna(subset=['next_over_runs', 'next_over_wickets'])
    
    return result_df

def process_ball_by_ball_data(deliveries_path='data/ball_to_ball/deliveries.csv',
                            matches_path='data/match_id/matches.csv',
                            output_path='cleaned_data/match_standardized.csv'):
    """Process ball-by-ball data to create the final dataset."""
    
    # Load data
    deliveries = pd.read_csv(deliveries_path)
    matches = pd.read_csv(matches_path)
    
    # Derive features
    processed_df = derive_features_from_ball_data(deliveries, matches)
    
    # Save processed data
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Dataset shape: {processed_df.shape}")
    print("\nFeatures available:")
    print(processed_df.columns.tolist())
    
    return processed_df



