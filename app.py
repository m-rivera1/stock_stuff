from flask import Flask, request, render_template, redirect, url_for
import stockModel


#-----------------------------------------------#
#              Initialize the app               #
#-----------------------------------------------#
app = Flask(__name__, static_folder="static")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = -1

#-----------------------------------------------#
#                   Global Variables            # 
#-----------------------------------------------#
ticker = ''

#-----------------------------------------------#
#                   Routes                      # 
#-----------------------------------------------#

@app.route("/")
def default():

    return render_template("index.html")

@app.route("/get-stock",methods=['GET','POST'])
def home():
    if request.method == 'POST':
        ticker = request.form['search']
        dates = stockModel.getStockDates(ticker)
        closeData = stockModel.getStockClose(ticker)
        companyName = stockModel.getCompanyName(ticker)


    return render_template("index.html", labels=dates, data=closeData, cName=companyName)

@app.route("/stock-analysis",methods=['GET','POST'])
def model():
    if request.method == 'POST':
        ticker = 'AMZN'
        dataset = stockModel.startModeling(ticker)
        # dates = stockModel.getStockDates(ticker)

        traindata = dataset["train"]
        train_dates = dataset["traindates"]
        actualdata = dataset["actual"]
        actual_dates = dataset['actualdates']
        predictdata = dataset["predict"]
        predict_dates = dataset['predictdates']

    
    return render_template("index.html", train=traindata,traindates=train_dates, actual=actualdata, actualdates=actual_dates,  predict=predictdata, predictdates=predict_dates)

if __name__ == '__main__':
    app.run(debug=True)