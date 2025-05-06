from flask import Flask, request, render_template, jsonify
import torch

app = Flask(__name__)

# Load your model (update device if required)
model_path = 'best_ipl_model.pth'
try:
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def index():
    # Serve the first model's input form by default
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
    try:
        # Get data from form
        over_num = float(request.form['over_num'])
        run_rate = float(request.form['run_rate'])
        req_run_rate = float(request.form['req_run_rate'])
        # target_left = float(request.form['target_left']) # This field is not used for 3-feature model

        # Prepare input tensor - adjust shape and features as per your model's actual requirements
        # Assuming model expects 3 features in a specific order
        input_data = torch.tensor([[over_num, run_rate, req_run_rate]], dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_data)
        
        prediction = output.tolist()
        # Assuming output is a single value or a list that can be easily displayed
        # Adjust formatting as needed
        prediction_display = f"{prediction[0]}" if isinstance(prediction, list) and len(prediction) > 0 else f"{prediction}"

        return render_template('result.html', prediction_output=prediction_display)
    except Exception as e:
        return render_template('result.html', prediction_output=f"Error processing request: {e}")

@app.route('/predict_model2', methods=['POST'])
def predict_model2():
    if model is None:
        return render_template('result.html', prediction_output="Error: Model not loaded.")
    try:
        # Get data from form
        wickets_fallen = float(request.form['wickets_fallen'])
        balls_remaining = float(request.form['balls_remaining'])
        venue_avg_score = float(request.form['venue_avg_score'])

        # Prepare input tensor - adjust shape and features as per your model's actual requirements
        # Assuming model expects 3 features in a specific order
        input_data = torch.tensor([[wickets_fallen, balls_remaining, venue_avg_score]], dtype=torch.float32)
        
        with torch.no_grad():
            output = model(input_data)
        
        prediction = output.tolist()
        # Assuming output is a single value or a list that can be easily displayed
        # Adjust formatting as needed
        prediction_display = f"{prediction[0]}" if isinstance(prediction, list) and len(prediction) > 0 else f"{prediction}"
        
        return render_template('result.html', prediction_output=prediction_display)
    except Exception as e:
        return render_template('result.html', prediction_output=f"Error processing request: {e}")

if __name__ == "__main__":
    app.run(debug=True)