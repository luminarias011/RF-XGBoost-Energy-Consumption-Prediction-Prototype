from flask import Flask, render_template, request, redirect, flash
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import time

app = Flask(__name__, static_url_path='/assets')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('upload_success.html', filename=filename)

@app.route('/simulate', methods=['POST'])
def simulate():
    if request.method == 'POST':
        start_time = time.time() 
        filename = request.form['filename']
        dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        data = pd.read_csv(dataset_path)
        
        # ? BASE
        rf_start_time = time.time()
        
        X_BASE = data[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']]
        Y_heating = data['Heating Load']
        Y_cooling = data['Cooling Load']

        X_train_heating, X_test_heating, y_train_heating, y_test_heating = train_test_split(X_BASE, Y_heating, test_size=0.2, random_state=20)
        X_train_cooling, X_test_cooling, y_train_cooling, y_test_cooling = train_test_split(X_BASE, Y_cooling, test_size=0.2, random_state=20)

        param_grid_rf = {
            'n_estimators': [100, 150],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }

        rf_model_heating = RandomForestRegressor(random_state=20)
        grid_search_rf_heating = GridSearchCV(rf_model_heating, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
        grid_search_rf_heating.fit(X_train_heating, y_train_heating)
        best_rf_model_heating = grid_search_rf_heating.best_estimator_

        rf_model_cooling = RandomForestRegressor(random_state=20)
        grid_search_rf_cooling = GridSearchCV(rf_model_cooling, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
        grid_search_rf_cooling.fit(X_train_cooling, y_train_cooling)
        best_rf_model_cooling = grid_search_rf_cooling.best_estimator_

        y_pred_rf_heating = best_rf_model_heating.predict(X_test_heating)
        y2_pred_rf_heating = best_rf_model_heating.predict(X_train_heating)

        y_pred_rf_cooling = best_rf_model_cooling.predict(X_test_cooling)
        y2_pred_rf_cooling = best_rf_model_cooling.predict(X_train_cooling)

        # Evaluate the model for heating load
        r2_rf_heating = r2_score(y_test_heating, y_pred_rf_heating)
        r22_rf_heating = r2_score(y_train_heating, y2_pred_rf_heating)
        mse_rf_heating = mean_squared_error(y_test_heating, y_pred_rf_heating)
        mse_rf_heating_traing = mean_squared_error(y_train_heating, y2_pred_rf_heating)
        rmse_rf_heating = np.sqrt(mse_rf_heating)
        rmse_rf_heating_train = np.sqrt(mse_rf_heating_traing)

        # Evaluate the model for cooling load
        r2_rf_cooling = r2_score(y_test_cooling, y_pred_rf_cooling)
        r22_rf_cooling = r2_score(y_train_cooling, y2_pred_rf_cooling)
        mse_rf_cooling = mean_squared_error(y_test_cooling, y_pred_rf_cooling)
        mse_rf_cooling_train = mean_squared_error(y_train_cooling, y2_pred_rf_cooling)
        rmse_rf_cooling = np.sqrt(mse_rf_cooling)
        rmse_rf_cooling_train = np.sqrt(mse_rf_cooling_train)
        
        
        # ! HEATING
        rf_heating_data = list(zip(y_test_heating.tolist(), y_pred_rf_heating.tolist()))
        # ? COOLING
        rf_cooling_data = list(zip(y_test_cooling.tolist(), y_pred_rf_cooling.tolist()))
        
        # ! Combination of Training and Cooling Metrics
        # ? R-SQUARED Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_r2_training = (r22_rf_heating * weight_heating + r22_rf_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # ? R-SQUARED Combinatin for TESTING
        weight_heating = 2
        weight_cooling = 1
        harmonic_r2_testing = (r2_rf_heating * weight_heating + r2_rf_cooling * weight_cooling) / (weight_cooling + weight_heating)
        
        # ? MSE Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_mse_training = (mse_rf_heating_traing * weight_heating + mse_rf_cooling_train * weight_cooling) / (weight_cooling + weight_heating)
       # ? MSE Combinatin for Testing
        weight_heating = 2
        weight_cooling = 1
        harmonic_mse_testing = (mse_rf_heating * weight_heating + mse_rf_cooling * weight_cooling) / (weight_cooling + weight_heating) 
        
        # ? RMSE Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_rmse_training = (rmse_rf_heating_train * weight_heating + rmse_rf_cooling_train * weight_cooling) / (weight_cooling + weight_heating) 
        # ? RMSE Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_rmse_testing = (rmse_rf_heating * weight_heating + rmse_rf_cooling * weight_cooling) / (weight_cooling + weight_heating) 
        
        rf_elapsed_time = time.time() - rf_start_time
        
        # ?
        # ! XGBoost
        # ?
        
        # Splitting data into features and target variables
        X = data[['Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area',
                'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution']]
        Y1 = data['Heating Load']
        Y2 = data['Cooling Load']

        xgb_start_time = time.time()
        # Splitting data into training and testing sets
        X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(X, Y1, Y2, test_size=0.2, random_state=20)

        # Best parameters for Heating Load
        best_params_heating = {'n_estimators': 100, 'learning_rate': 0.02, 'max_depth': 70, 'min_child_weight': 2}
        best_model_heating = xgb.XGBRegressor(objective='reg:squarederror', random_state=20, **best_params_heating)
        best_model_heating.fit(X_train, Y1_train)

        # Predictions for Heating Load
        Y1_pred = best_model_heating.predict(X_test)

        # Best parameters for Cooling Load
        best_params_cooling = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 70, 'min_child_weight': 2}
        best_model_cooling = xgb.XGBRegressor(objective='reg:squarederror', random_state=20, **best_params_cooling)
        best_model_cooling.fit(X_train, Y2_train)

        # Predictions for Cooling Load
        Y2_pred = best_model_cooling.predict(X_test)

        # !
        # Training and testing R-squared for Heating Load
        r2_train_heating = best_model_heating.score(X_train, Y1_train)
        r2_test_heating = best_model_heating.score(X_test, Y1_test)
        # Calculating MSE and RMSE for Heating Load
        mse_train_heating = mean_squared_error(Y1_train, best_model_heating.predict(X_train))
        rmse_train_heating = np.sqrt(mse_train_heating)
        mse_test_heating = mean_squared_error(Y1_test, Y1_pred)
        rmse_test_heating = np.sqrt(mse_test_heating)

        # Training and testing R-squared for Cooling Load
        r2_train_cooling = best_model_cooling.score(X_train, Y2_train)
        r2_test_cooling = best_model_cooling.score(X_test, Y2_test)
        # Calculating MSE and RMSE for Cooling Load
        mse_train_cooling = mean_squared_error(Y2_train, best_model_cooling.predict(X_train))
        rmse_train_cooling = np.sqrt(mse_train_cooling)
        mse_test_cooling = mean_squared_error(Y2_test, Y2_pred)
        rmse_test_cooling = np.sqrt(mse_test_cooling)
        
        # ! HEATING
        xgb_heating_data = list(zip(Y1_test.tolist(), Y1_pred.tolist()))
        # ? COOLING
        xgb_cooling_data = list(zip(Y2_test.tolist(), Y2_pred.tolist()))

        # ! Combination of Training and Cooling Metrics
        # ? R-SQUARED Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_r2_XGB_training = (r2_train_heating * weight_heating + r2_train_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # ? R-SQUARED Combinatin for TESTING
        weight_heating = 1
        weight_cooling = 2
        harmonic_r2_XGB_testing = (r2_test_heating * weight_heating + r2_test_cooling * weight_cooling) / (weight_cooling + weight_heating)

        # ? MSE Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_mse_XGB_training = (mse_train_heating * weight_heating + mse_train_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # ? MSE Combinatin for TESTING
        weight_heating = 2
        weight_cooling = 1
        harmonic_mse_XGB_testing = (mse_test_heating * weight_heating + mse_test_cooling * weight_cooling) / (weight_cooling + weight_heating)

        # ? RMSE Combinatin for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_rmse_XGB_training = (rmse_train_heating * weight_heating + rmse_train_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # ? RMSE Combinatin for TESTING
        weight_heating = 2
        weight_cooling = 1
        harmonic_rmse_XGB_testing = (rmse_test_heating * weight_heating + rmse_test_cooling * weight_cooling) / (weight_cooling + weight_heating)

        xgb_elapsed_time = time.time() - xgb_start_time
        
        # ? Hybrid
        # ! HYBRID MODEL
        # ? Hybrid
        
        hybrid_start_time = time.time()
        imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        data_imputed['Volume'] = data_imputed['Surface Area'] * data_imputed['Overall Height']
        data_imputed['Wall/Roof Ratio'] = data_imputed['Wall Area'] / data_imputed['Roof Area']

        data_final = data_imputed

        X = data_final.drop(['Heating Load', 'Cooling Load'], axis=1)
        Y1 = data_final['Heating Load']
        Y2 = data_final['Cooling Load']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, Y1_train, Y1_test, Y2_train, Y2_test = train_test_split(X_scaled, Y1, Y2, test_size=0.2, random_state=20)

        rf_model_heating = RandomForestRegressor(random_state=20, n_estimators=500, max_depth=70, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
        rf_model_heating.fit(X_train, Y1_train)

        rf_model_cooling = RandomForestRegressor(random_state=20, n_estimators=500, max_depth=70, min_samples_split=3, min_samples_leaf=1, max_features='sqrt', bootstrap=False)
        rf_model_cooling.fit(X_train, Y2_train)

        xgb_model_heating = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', subsample=0.8, random_state=20, n_estimators=2000, max_depth=70, learning_rate=0.05, min_child_weight=2, gamma=0, colsample_bytree=0.6)
        xgb_model_heating.fit(X_train, Y1_train)

        xgb_model_cooling = xgb.XGBRegressor(objective='reg:squarederror', eval_metric='rmse', subsample=0.8, random_state=20, n_estimators=2000, max_depth=70, learning_rate=0.05, min_child_weight=2, gamma=0, colsample_bytree=0.6)
        xgb_model_cooling.fit(X_train, Y2_train)

        heating_models = [('Random Forest', rf_model_heating), ('XGBoost', xgb_model_heating)]
        heating_ensemble = VotingRegressor(heating_models)
        heating_ensemble.fit(X_train, Y1_train)

        cooling_models = [('Random Forest', rf_model_cooling), ('XGBoost', xgb_model_cooling)]
        cooling_ensemble = VotingRegressor(cooling_models)
        cooling_ensemble.fit(X_train, Y2_train)

        Y1_pred_train_ensemble = heating_ensemble.predict(X_train)
        Y1_pred_test_ensemble = heating_ensemble.predict(X_test)

        Y2_pred_train_ensemble = cooling_ensemble.predict(X_train)
        Y2_pred_test_ensemble = cooling_ensemble.predict(X_test)

        # Evaluation metrics for Heating Load
        r2_train_ensemble_heating = r2_score(Y1_train, Y1_pred_train_ensemble)
        r2_test_ensemble_heating = r2_score(Y1_test, Y1_pred_test_ensemble)
        mse_train_ensemble_heating = mean_squared_error(Y1_train, Y1_pred_train_ensemble)
        mse_test_ensemble_heating = mean_squared_error(Y1_test, Y1_pred_test_ensemble)
        rmse_train_ensemble_heating = np.sqrt(mse_train_ensemble_heating)
        rmse_test_ensemble_heating = np.sqrt(mse_test_ensemble_heating)

        # Evaluation metrics for Cooling Load
        r2_train_ensemble_cooling = r2_score(Y2_train, Y2_pred_train_ensemble)
        r2_test_ensemble_cooling = r2_score(Y2_test, Y2_pred_test_ensemble)
        mse_train_ensemble_cooling = mean_squared_error(Y2_train, Y2_pred_train_ensemble)
        mse_test_ensemble_cooling = mean_squared_error(Y2_test, Y2_pred_test_ensemble)
        rmse_train_ensemble_cooling = np.sqrt(mse_train_ensemble_cooling)
        rmse_test_ensemble_cooling = np.sqrt(mse_test_ensemble_cooling)
        
        
        # ! HEATING
        hybrid_heating_data = list(zip(Y1_train.tolist(), Y1_pred_train_ensemble.tolist()))
        # ? COOLING
        hybrid_cooling_data = list(zip(Y2_train.tolist(), Y2_pred_train_ensemble.tolist()))
        
        # ! HEATING TEST
        hybrid_heating_data_test = list(zip(Y1_test.tolist(), Y1_pred_test_ensemble.tolist()))
        # ? COOLING TEST
        hybrid_cooling_data_test = list(zip(Y2_test.tolist(), Y2_pred_test_ensemble.tolist()))
        

        # ! Combining training and testing Metrics

        # ? R-Squared for TRAINING
        weight_heating = 2
        weight_cooling = 1
        harmonic_r2_mean_training = (r2_train_ensemble_heating * weight_heating + r2_train_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # harmonic_r2_mean_training = 2 / ((1 / r2_train_ensemble_heating) + (1 / r2_train_ensemble_cooling))
        # ? R-Squared for TESTING
        weight_heating = 2
        weight_cooling = 1
        harmonic_r2_mean_testing = (r2_test_ensemble_heating * weight_heating + r2_test_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)
        # harmonic_r2_mean_testing = 2 / ((1 / r2_test_ensemble_heating) + (1 / r2_test_ensemble_cooling))

        # ? MSE for TRAINING
        weight_heating = 2
        weight_cooling = 1
        weighted_average_mse_training = (mse_train_ensemble_heating * weight_heating + mse_train_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)

        # ? MSE for Testing
        weight_heating = 2
        weight_cooling = 1
        weighted_average_mse_testing = (mse_test_ensemble_heating * weight_heating + mse_test_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)

        # ? RMSE for Testing
        weight_heating = 2
        weight_cooling = 1
        weighted_average_rmse_training = (rmse_train_ensemble_heating * weight_heating + rmse_train_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)

        # ? RMSE for Testing
        weight_heating = 2
        weight_cooling = 1
        weighted_average_rmse_testing = (rmse_test_ensemble_heating * weight_heating + rmse_test_ensemble_cooling * weight_cooling) / (weight_cooling + weight_heating)
        
        hybrid_elapsed_time = time.time() - hybrid_start_time
        
        total_elapsed_time = time.time() - start_time
        
        return render_template('simulation_result.html', 
                                    harmonic_r2_training=harmonic_r2_training,
                                    harmonic_r2_testing=harmonic_r2_testing,
                                    harmonic_mse_training=harmonic_mse_training,
                                    harmonic_mse_testing=harmonic_mse_testing,
                                    harmonic_rmse_training=harmonic_rmse_training,
                                    harmonic_rmse_testing=harmonic_rmse_testing,
                                    # *
                                    rf_heating_data=rf_heating_data,
                                    rf_cooling_data=rf_cooling_data,
                                    # ?
                                    harmonic_r2_XGB_training=harmonic_r2_XGB_training,
                                    harmonic_r2_XGB_testing=harmonic_r2_XGB_testing,
                                    harmonic_mse_XGB_training=harmonic_mse_XGB_training,
                                    harmonic_mse_XGB_testing=harmonic_mse_XGB_testing,
                                    harmonic_rmse_XGB_training=harmonic_rmse_XGB_training,
                                    harmonic_rmse_XGB_testing=harmonic_rmse_XGB_testing,
                                    # *
                                    xgb_heating_data=xgb_heating_data,
                                    xgb_cooling_data=xgb_cooling_data,
                                    # ?
                                    harmonic_r2_mean_training=harmonic_r2_mean_training,
                                    harmonic_r2_mean_testing=harmonic_r2_mean_testing,
                                    weighted_average_mse_training=weighted_average_mse_training,
                                    weighted_average_mse_testing=weighted_average_mse_testing,
                                    weighted_average_rmse_training=weighted_average_rmse_training,
                                    weighted_average_rmse_testing=weighted_average_rmse_testing,
                                    # *
                                    hybrid_heating_data=hybrid_heating_data,
                                    hybrid_cooling_data=hybrid_cooling_data,
                                    # !
                                    hybrid_heating_data_test=hybrid_heating_data_test,
                                    hybrid_cooling_data_test=hybrid_cooling_data_test,
                                    # ?
                                    rf_elapsed_time=rf_elapsed_time,
                                    xgb_elapsed_time=xgb_elapsed_time,
                                    hybrid_elapsed_time=hybrid_elapsed_time,
                                    total_elapsed_time=total_elapsed_time
                              )

if __name__ == '__main__':
    app.run(debug=True)
