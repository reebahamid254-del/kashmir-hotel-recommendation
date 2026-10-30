from flask import Flask, render_template, request
import numpy as np
import pickle as pkl

app = Flask(__name__)

# Load the trained model and encoder
with open('decision_tree.pkl', 'rb') as f:
    model = pkl.load(f)
with open('ohe.pkl', 'rb') as f:
    ohe = pkl.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # 1. Extract and convert numerical / binary features
            user_budget_total_inr = float(request.form['user_budget_total_inr'])
            user_daily_budget_inr = float(request.form['user_daily_budget_inr'])
            stay_days = float(request.form['stay_days'])
            family_count = float(request.form['family_count'])
            adults_count = float(request.form['adults_count'])
            children_count = float(request.form['children_count'])
            seniors_count = float(request.form['seniors_count'])
            rooms_required = float(request.form['rooms_required'])
            
            requires_central_heating = float(request.form['requires_central_heating'])
            requires_scenic_view = float(request.form['requires_scenic_view'])
            star_category = float(request.form['star_category'])
            google_rating = float(request.form['google_rating'])
            review_count = float(request.form['review_count'])
            base_price_per_night_inr = float(request.form['base_price_per_night_inr'])
            seasonal_room_rate_inr = float(request.form['seasonal_room_rate_inr'])
            estimated_total_stay_cost_inr = float(request.form['estimated_total_stay_cost_inr'])
            budget_fit_ratio = float(request.form['budget_fit_ratio'])
            max_guests_per_room = float(request.form['max_guests_per_room'])
            
            has_central_heating_bukhari = float(request.form['has_central_heating_bukhari'])
            has_power_backup_24x7 = float(request.form['has_power_backup_24x7'])
            has_mountain_or_lake_view = float(request.form['has_mountain_or_lake_view'])
            has_wazwan_kitchen = float(request.form['has_wazwan_kitchen'])
            
            distance_to_sxr_airport_km = float(request.form['distance_to_sxr_airport_km'])
            distance_to_landmark_km = float(request.form['distance_to_landmark_km'])
            latitude = float(request.form['latitude'])
            longitude = float(request.form['longitude'])
            altitude_m = float(request.form['altitude_m'])
            suitability_score = float(request.form['suitability_score'])

            # 2. Extract categorical features
            travel_season = request.form['travel_season']
            trip_purpose = request.form['trip_purpose']
            preferred_property_type = request.form['preferred_property_type']
            dietary_preference = request.form['dietary_preference']
            district = request.form['district']
            property_type = request.form['property_type']

            # 3. Assemble numerical array in expected order
            nums = [[
                user_budget_total_inr, user_daily_budget_inr, stay_days, family_count,
                adults_count, children_count, seniors_count, rooms_required,
                requires_central_heating, requires_scenic_view, star_category,
                google_rating, review_count, base_price_per_night_inr,
                seasonal_room_rate_inr, estimated_total_stay_cost_inr, budget_fit_ratio,
                max_guests_per_room, has_central_heating_bukhari, has_power_backup_24x7,
                has_mountain_or_lake_view, has_wazwan_kitchen, distance_to_sxr_airport_km,
                distance_to_landmark_km, latitude, longitude, altitude_m, suitability_score
            ]]

            # 4. Encode categorical features using ohe.pkl
            cat = ohe.transform([[
                travel_season, trip_purpose, preferred_property_type,
                dietary_preference, district, property_type
            ]])

            # 5. Combine and predict
            input_data = np.hstack([nums, cat])
            pred_result = model.predict(input_data)
            prediction = pred_result[0]

        except Exception as e:
            error_message = str(e)

    return render_template('index.html', prediction=prediction, error=error_message)

if __name__ == '__main__':
    app.run(debug=True)