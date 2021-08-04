from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle

# load the model from disk
predictor_model_mobile=pickle.load(open('XGBoost_model.pkl', 'rb'))
application = Flask(__name__)

@application.route('/')
def home():
	return render_template('home.html')

@application.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Yearly_avg_view_on_travel_page = float(request.form['Yearly_avg_view_on_travel_page'])
        total_likes_on_outstation_checkin_given = float(request.form['total_likes_on_outstation_checkin_given'])
        yearly_avg_Outstation_checkins = float(request.form['yearly_avg_Outstation_checkins'])
        member_in_family = float(request.form['member_in_family'])
        preferred_location_type = float(request.form['preferred_location_type'])
        Yearly_avg_comment_on_travel_page =  float(request.form['Yearly_avg_comment_on_travel_page'])
        total_likes_on_outofstation_checkin_received =  float(request.form['total_likes_on_outofstation_checkin_received'])
        week_since_last_outstation_checkin = float(request.form['week_since_last_outstation_checkin'])
        is_following_company_page = request.form['following_company_page']

        if (is_following_company_page == 'Yes') or (is_following_company_page == 'yes'):
            following_company_page = 1.0
        else:
            following_company_page = 0.0

        montly_avg_comment_on_company_page = float(request.form['montly_avg_comment_on_company_page'])
        is_working_flag = request.form['working_flag']

        if (is_working_flag == 'Yes') or (is_working_flag == 'yes'):
            working_flag = 1.0
        else:
            working_flag = 0.0

        travelling_network_rating = float(request.form['travelling_network_rating'])
        is_Adult_flag = request.form['Adult_flag']

        if (is_Adult_flag == 'Yes') or (is_Adult_flag == 'yes'):
            Adult_flag = 1.0
        else:
            Adult_flag = 0.0

        Daily_Avg_mins_spend_on_traveling_page = float(request.form['Daily_Avg_mins_spend_on_traveling_page'])

        data = np.array([[Yearly_avg_view_on_travel_page, total_likes_on_outstation_checkin_given, yearly_avg_Outstation_checkins, member_in_family, preferred_location_type, Yearly_avg_comment_on_travel_page, total_likes_on_outofstation_checkin_received, week_since_last_outstation_checkin, following_company_page, montly_avg_comment_on_company_page, working_flag, travelling_network_rating, Adult_flag, Daily_Avg_mins_spend_on_traveling_page]])
        my_prediction = predictor_model_mobile.predict(data)

        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	application.run(debug=True)