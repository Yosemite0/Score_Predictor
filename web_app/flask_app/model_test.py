from flask import Flask, request, render_template
import torch
import os
import math # For calculations like floor, round

app = Flask(__name__)

# --- Global variables for model and preprocessing objects ---
model = None
model_params = None
feature_info = None
# TODO: These need to be loaded or recreated based on your training process.
# scaler = None  # Should be a fitted sklearn.preprocessing.StandardScaler
# team_encoder = None # Should be a fitted sklearn.preprocessing.LabelEncoder

# Load your model and configuration
model_path = '../../best_ipl_model.pth'
if os.path.exists(model_path):
    try:
        # Ensure model_definition.py is in the Python path or same directory
        from model_definition import IPLScorePredictor #, CombinedIPLLoss # If loss is needed

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint is not a dictionary. Ensure it was saved with model_params and feature_info.")

        model_params = checkpoint.get('model_params')
        feature_info = checkpoint.get('feature_info')
        
        if not model_params:
            raise ValueError("Model parameters (model_params) not found in checkpoint.")
        # feature_info is useful for knowing numerical feature names and team classes
        if not feature_info:
            app.logger.warning("Feature info (feature_info) not found in checkpoint. Numerical feature names and team classes might be unavailable for robust preprocessing.")
            # Provide a default or fallback if absolutely necessary, though it's best to ensure it's saved.
            feature_info = {
                'numerical_features': ['over', 'curr_run_rate', 'req_runrate', 'target_left', 'weighted_run_rate'], 
                'team_classes': [] # This would prevent team_encoder from being properly fitted
            }

        model = IPLScorePredictor(**model_params)
        
        # Load model state dictionary
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint: # Some older checkpoints might use 'state_dict'
             model.load_state_dict(checkpoint["state_dict"])
        else:
            # If the checkpoint root is the state_dict itself (less common for complex saves)
            # This might indicate an issue with how the checkpoint was saved or loaded.
            app.logger.warning("Attempting to load checkpoint root as model_state_dict. Ensure checkpoint structure is as expected.")
            model.load_state_dict(checkpoint)
        
        model.eval()
        app.logger.info("Model loaded successfully.")
        app.logger.info(f"Model Parameters: {model_params}")
        app.logger.info(f"Feature Info: {feature_info}")

        # --- TODO: Initialize and load/fit scaler and team_encoder here ---
        # Example for team_encoder if team_classes are available:
        # from sklearn.preprocessing import LabelEncoder
        # team_encoder = LabelEncoder()
        # if feature_info and 'team_classes' in feature_info and feature_info['team_classes']:
        #     team_encoder.fit(feature_info['team_classes'])
        #     app.logger.info(f"Team encoder fitted with classes: {team_encoder.classes_}")
        # else:
        #     app.logger.warning("Team classes not available from feature_info to fit LabelEncoder. Team name encoding will be placeholder.")

        # Example for scaler (you'd need to have saved mean_ and scale_ or the scaler object itself)
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # if 'scaler_params' in checkpoint and checkpoint['scaler_params'].get('mean_') is not None:
        #     scaler.mean_ = torch.tensor(checkpoint['scaler_params']['mean_']).numpy() # Ensure numpy array
        #     scaler.scale_ = torch.tensor(checkpoint['scaler_params']['scale_']).numpy() # Ensure numpy array
        #     app.logger.info("Scaler loaded with mean and scale.")
        # else:
        #     app.logger.warning("Scaler parameters (mean_, scale_) not found in checkpoint. Numerical features will use raw values (INACCURATE).")
        # --- End of TODO for scaler and team_encoder ---

    except Exception as e:
        app.logger.error(f"Error loading model or configuration from {model_path}: {e}", exc_info=True)
        model = None
else:
    app.logger.error(f"Error: Model file not found at {model_path}")
    model = None

@app.route('/')
def index():
    return render_template('model1.html')

@app.route('/model1_form')
def model1_form():
    return render_template('model1.html')

@app.route('/model2_form')
def model2_form():
    return render_template('model2.html')

@app.route('/predict_model1', methods=['POST'])
def predict_model1():
    if model is None:
        return render_template('result.html', prediction_output="Error: Model not loaded.")
    if model_params is None: # feature_info might be partially missing but model_params is essential
        return render_template('result.html', prediction_output="Error: Model configuration (model_params) not loaded.")

    try:
        # Get data from form
        innings_str = request.form['innings']
        over_num_raw = float(request.form['over_num']) # e.g., 5.2 for 5 overs, 2 balls
        current_score = float(request.form['current_score'])
        batting_team_name = request.form['batting_team']
        bowling_team_name = request.form['bowling_team']
        
        # --- Derive numerical features ---
        # 1. Over
        over_val = over_num_raw 

        # 2. Current Run Rate
        over_major = math.floor(over_num_raw)
        over_minor_balls = round((over_num_raw - over_major) * 10)
        actual_overs_bowled = 0
        if over_major > 0 or over_minor_balls > 0: # Avoid division by zero if 0.0 overs
            actual_overs_bowled = over_major + (over_minor_balls / 6.0)
        
        curr_run_rate_val = 0.0
        if actual_overs_bowled > 0:
            curr_run_rate_val = current_score / actual_overs_bowled
        
        # 3. Required Run Rate & 4. Target Left (conditional on innings)
        req_run_rate_val = 0.0  # Default for 1st innings
        target_left_val = 0.0   # Default for 1st innings (or a high value if model expects that)
        
        if innings_str == '2':
            target_score_str = request.form.get('target_score')
            req_run_rate_form_str = request.form.get('req_run_rate') # User might provide this directly
            
            if not target_score_str: # Target score is essential for 2nd innings
                return render_template('result.html', prediction_output="Error: Target score needed for 2nd innings.")
            
            target_score = float(target_score_str)
            target_left_val = max(0, target_score - current_score) # Target left cannot be negative

            # Calculate req_run_rate if not provided or to verify
            balls_bowled = over_major * 6 + over_minor_balls
            balls_remaining_total = 120 # Assuming T20, 20 overs * 6 balls
            balls_remaining_inning = balls_remaining_total - balls_bowled
            
            if balls_remaining_inning > 0:
                calculated_req_run_rate = (target_left_val / balls_remaining_inning) * 6
                # Use form value if provided and seems reasonable, otherwise use calculated
                if req_run_rate_form_str:
                    req_run_rate_val = float(req_run_rate_form_str)
                else:
                    req_run_rate_val = calculated_req_run_rate
            else: # No balls remaining, req_run_rate is theoretically infinite if target not met
                req_run_rate_val = 99 if target_left_val > 0 else 0 # Assign a high value or 0
        
        # 5. Weighted Run Rate (Using current run rate as a proxy - THIS IS A LIMITATION)
        # For accurate prediction, this feature should ideally be calculated based on historical over data
        # or be an input if the model was trained with it as a direct input.
        weighted_run_rate_val = curr_run_rate_val 
        app.logger.warning("Using 'current_run_rate' as a proxy for 'weighted_run_rate'. This is a simplification.")
        
        # --- Assemble numerical features in the order expected by the scaler/model ---
        # The order should match feature_info['numerical_features'] if available and used for scaler fitting
        # Default assumed order: ['over', 'curr_run_rate', 'req_runrate', 'target_left', 'weighted_run_rate']
        raw_numerical_features_dict = {
            'over': over_val,
            'curr_run_rate': curr_run_rate_val,
            'req_runrate': req_run_rate_val,
            'target_left': target_left_val,
            'weighted_run_rate': weighted_run_rate_val
        }
        
        # Order them based on feature_info if possible
        numerical_feature_names_from_info = feature_info.get('numerical_features', list(raw_numerical_features_dict.keys()))
        raw_numerical_features = [raw_numerical_features_dict[name] for name in numerical_feature_names_from_info]

        # --- Placeholder for Scaling ---
        # TODO: Scale numerical_features using the loaded 'scaler'
        # if scaler:
        #     scaled_numerical_features = scaler.transform([raw_numerical_features])[0]
        # else:
        app.logger.warning("Scaler not loaded or not applied. Using raw numerical features (WILL LIKELY LEAD TO INACCURATE PREDICTIONS).")
        scaled_numerical_features = raw_numerical_features # Using raw values as a fallback

        # --- Placeholder for Team Encoding ---
        # TODO: Encode team names using the loaded 'team_encoder'
        # if team_encoder:
        #     try:
        #         batting_team_encoded = float(team_encoder.transform([batting_team_name])[0])
        #         bowling_team_encoded = float(team_encoder.transform([bowling_team_name])[0])
        #     except ValueError as ve:
        #         app.logger.error(f"Team name not found in encoder: {ve}. Using placeholder.")
        #         batting_team_encoded = 0.0 # Placeholder for unknown team
        #         bowling_team_encoded = 0.0 # Placeholder for unknown team
        # else:
        app.logger.warning("Team encoder not loaded or not applied. Using placeholder team encodings (0.0) (WILL LIKELY LEAD TO INACCURATE PREDICTIONS).")
        batting_team_encoded = 0.0 # Placeholder
        bowling_team_encoded = 0.0 # Placeholder

        all_features = list(scaled_numerical_features) + [batting_team_encoded, bowling_team_encoded]
        
        expected_input_size = model_params.get('input_size')
        if len(all_features) != expected_input_size:
            error_msg = f"Feature count mismatch. Model expected {expected_input_size}, but got {len(all_features)} features."
            app.logger.error(error_msg)
            return render_template('result.html', prediction_output=f"Error: {error_msg}")

        # Prepare input tensor: (batch_size=1, seq_len=1, num_features)
        # The model's LSTM expects a sequence, so unsqueeze(1) adds the seq_len dimension.
        input_tensor = torch.tensor([all_features], dtype=torch.float32).unsqueeze(1)
        seq_lengths = torch.tensor([1], dtype=torch.long) # Sequence length is 1 for this single prediction

        with torch.no_grad():
            output = model(input_tensor, seq_lengths)
        
        prediction_list = output.tolist()
        
        if not prediction_list or not prediction_list[0] or len(prediction_list[0]) != model_params.get('output_size', 2):
             error_msg = f"Unexpected model output format: {prediction_list}. Expected output size {model_params.get('output_size', 2)}."
             app.logger.error(error_msg)
             return render_template('result.html', prediction_output=f"Error: {error_msg}")

        # Get raw predictions
        predicted_runs = prediction_list[0][0]
        wicket_logits = prediction_list[0][1]
        
        # Apply reasonable run clamping (max 36 runs per over - 6 sixes)
        predicted_runs = max(0, min(36, predicted_runs))
        
        # Adjust prediction based on match context
        if curr_run_rate_val > 10:  # High scoring match
            predicted_runs *= 1.2  # Increase prediction by 20%
        
        # Round the prediction for display
        predicted_runs = round(predicted_runs)
        
        wicket_probability = torch.sigmoid(torch.tensor(wicket_logits)).item()

        prediction_display = (f"Predicted Final Runs: {predicted_runs:.2f}<br>"
                              f"Predicted Wicket Probability (for this state leading to final): {wicket_probability:.2%}")

        return render_template('result.html', prediction_output=prediction_display)
    except Exception as e:
        app.logger.error(f"Error processing /predict_model1: {e}", exc_info=True)
        return render_template('result.html', prediction_output=f"Error processing request: {e}")

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    # This route uses different input features (wickets_fallen, balls_remaining, venue_avg_score)
    # which are not directly compatible with the 7-feature IPLScorePredictor model loaded.
    # This will likely result in errors or incorrect predictions if the same model is used.
    if model is None:
        return render_template('result.html', prediction_output="Error: Model not loaded.")
    
    app.logger.warning("/predict_model2 is called, but it's designed for a different feature set than the loaded 7-feature model.")
    # For demonstration, if you were to try and use it (it would be incorrect):
    # try:
    #     wickets_fallen = float(request.form['wickets_fallen'])
    #     balls_remaining = float(request.form['balls_remaining'])
    #     venue_avg_score = float(request.form['venue_avg_score'])
    #     # This is NOT the correct input for the 7-feature model.
    #     # input_data = torch.tensor([[wickets_fallen, balls_remaining, venue_avg_score, 0, 0, 0, 0]], dtype=torch.float32).unsqueeze(1) 
    #     # ... model call ...
    # except Exception as e:
    #    return render_template('result.html', prediction_output=f"Error in predict_model2: {e}")
        
    return render_template('result.html', prediction_output="Notice: predict_model2 uses a different feature set. The currently loaded model is for 7 specific features (over, run rates, teams, etc.). This route may not provide meaningful predictions with the current model.")

if __name__ == "__main__":
    # Basic logging for Flask
    import logging
    logging.basicConfig(level=logging.INFO) # Set root logger level
    # Ensure Flask app logger also respects this level or set its own
    app.logger.setLevel(logging.INFO) 
    
    # Remove default Flask handler if you want to avoid duplicate logs with basicConfig
    # from flask.logging import default_handler
    # app.logger.removeHandler(default_handler)

    # Add a stream handler to see logs in console
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    # app.logger.addHandler(handler) # basicConfig might already set up a stream handler for root

    app.run(debug=True)