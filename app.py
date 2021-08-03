from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle

# load the model from disk
predictor_model_mobile=pickle.load(open('/Users/sunandakumarghosh/Desktop/Social_Media_Tourism/XGBoost_model.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    # df=pd.read_csv('real_2018.csv')
    # my_prediction=loaded_model.predict(df.iloc[:,:-1].values)
    # my_prediction=my_prediction.tolist()

    if request.method == 'POST':
        Yearly_avg_view_on_travel_page = float(request.form['Yearly_avg_view_on_travel_page'])
        total_likes_on_outstation_checkin_given = float(request.form['total_likes_on_outstation_checkin_given'])
        yearly_avg_Outstation_checkins = float(request.form['yearly_avg_Outstation_checkins'])
        member_in_family = float(request.form['member_in_family'])
        preferred_location_type = float(request.form['preferred_location_type'])
        Yearly_avg_comment_on_travel_page =  float(request.form['Yearly_avg_comment_on_travel_page'])
        total_likes_on_outofstation_checkin_received =  float(request.form['total_likes_on_outofstation_checkin_received'])
        week_since_last_outstation_checkin = float(request.form['week_since_last_outstation_checkin'])
        following_company_page = float(request.form['following_company_page'])
        montly_avg_comment_on_company_page = float(request.form['montly_avg_comment_on_company_page'])
        working_flag = float(request.form['working_flag'])
        travelling_network_rating = float(request.form['travelling_network_rating'])
        Adult_flag = float(request.form['Adult_flag'])
        Daily_Avg_mins_spend_on_traveling_page = float(request.form['Daily_Avg_mins_spend_on_traveling_page'])

        data = np.array([[Yearly_avg_view_on_travel_page, total_likes_on_outstation_checkin_given, yearly_avg_Outstation_checkins, member_in_family, preferred_location_type, Yearly_avg_comment_on_travel_page, total_likes_on_outofstation_checkin_received, week_since_last_outstation_checkin, following_company_page, montly_avg_comment_on_company_page, working_flag, travelling_network_rating, Adult_flag, Daily_Avg_mins_spend_on_traveling_page]])
        my_prediction = predictor_model_mobile.predict(data)

        return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)