from flask import Flask
from flask import render_template
from flask import request, redirect, url_for
import pandas as pd
import cohere
import json
from algosdk.v2client import algod
from algosdk import account, mnemonic, kmd
from algosdk import transaction
# from algosdk.future.transaction import AssetConfigTxn, AssetTransferTxn, AssetFreezeTxn
from cohere.classify import Example
from db import DB, City, Hospital, Review, User
from models.NLP.wordcloud1 import generate_wordcloud
import requests
import sqlite3

co = cohere.Client('xprvLYyVPF4XKxRqRUb1ADD9TiQ8ATDCKN6kuqVk')

app=Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///hospitalReview.db'
DB.init_app(app)
# SQLALCHEMY_TRACK_MODIFICATIONS = False

VERIFICATION_TOKEN = ''

@app.route('/', methods=['GET', 'POST'])
def index():
    conn = sqlite3.connect('hospitalReview.db')
    c = conn.cursor()
                
    hospitals=c.execute('SELECT * FROM hospital').fetchall()
    hospital_names=[hospital[1] for hospital in hospitals]
    hospital_ids=[hospital[0] for hospital in hospitals]
    #get avgRatings from db using review table
    avgRatings=[]
    for hospital in hospitals:
        reviews=c.execute('SELECT * FROM review WHERE hospital_id=?',(hospital[0],)).fetchall()
        if len(reviews)==0:
            avgRatings.append(0)
        else:
            avgRatings.append(round(sum([review[5] for review in reviews])/len(reviews),2))
    #get city of hospitals from db using city table
    cities=[]
    for hospital in hospitals:
        city=c.execute('SELECT * FROM city WHERE id=?',(hospital[2],)).fetchone()
        cities.append(city[1])
    #get speciality of hospitals from db using hospital table
    specialities=[]
    for hospital in hospitals:
        specialities.append(hospital[3])
    
    if request.method == 'POST':
        hosp=request.form['hospitalName']
        #get hospital city,speciality,reviews from the database
        hospital=c.execute('SELECT * FROM hospital WHERE name=?',(hosp,)).fetchone()
        city=c.execute('SELECT * FROM city WHERE id=?',(hospital[2],)).fetchone()
        reviews=c.execute('SELECT * FROM review WHERE hospital_id=?',(hospital[0],)).fetchall()

        #get the reviews and put them in a list
        reviewList=[]
        ratingList=[]
        dateList=[]
        confidenceList=[]
        for review in reviews:
            reviewList.append(review[3])
            ratingList.append(float(review[5]))
            dateList.append(review[4])
            confidenceList.append(review[6])
        print(ratingList)
        avgRating=sum(ratingList)/len(ratingList)
        avgRating=round(avgRating,2)
        #get the hospital name and city
        hospitalName=hospital[1]
        cityName=city[1]
        #get the speciality
        speciality=hospital[3]
        #get the wordcloud
        generate_wordcloud(reviewList)

        #convert ratings to str
        for i in range(len(ratingList)):
            ratingList[i]=str(ratingList[i])
    
        # join the reviews by \n
        reviewString='||'.join(reviewList)
        ratingString='||'.join(ratingList)
        dateString='||'.join(dateList)
        confidenceString='||'.join(confidenceList)
        #redirect to transaction route
        return redirect(url_for('addReview', hospitalName=hospitalName, cityName=cityName, speciality=speciality,reviewList=reviewString,avgRating=avgRating, ratingList=ratingString, dateList=dateString, confidenceList=confidenceString))

    return render_template('index.html',hospital_names=hospital_names,avgRatings=avgRatings,cities=cities,specialities=specialities)

@app.route('/addReview', methods=['GET', 'POST'])
def addReview():
    if request.method=='POST':
        return redirect(url_for('txn'))
    return render_template('index2.html', hospitalName=request.args.get('hospitalName'), cityName=request.args.get('cityName'), speciality=request.args.get('speciality'), reviewList=request.args.get('reviewList'),ratingList=request.args.get('ratingList'),avgRating=request.args.get('avgRating'),dateList=request.args.get('dateList'), confidenceList=request.args.get('confidenceList'))
    

@app.route('/transaction', methods=['GET', 'POST'])
def txn():
    if request.method == 'POST':
        review = request.form['review']
        pp = request.form['pp']
        hospitalName = request.form['hospitalName']
        rating=request.form['rating']
        bill=request.files['bill']
        bill.save('bill.jpg')
        BILL_VERIFICATION_TOKEN=''
        url = 'https://app.nanonets.com/api/v2/OCR/Model/9d42c87a-d7bf-4e29-902f-c367ce5d5da8/LabelFile/'
        data = {'file': open('bill.jpg', 'rb')}
        response = requests.post(url, auth=requests.auth.HTTPBasicAuth('VMKD_k5J8_RNlzO9chHTYbnAjvuHbn63', ''), files=data)
        response = response.json()
        prediction=response['result'][0]['prediction'][0]['ocr_text']
        print(prediction)
        if prediction.lower()==hospitalName.lower():
            BILL_VERIFICATION_TOKEN='Verified'
            algod_address = "https://testnet-algorand.api.purestake.io/ps2"
            algod_token = ""
            headers = {
                "X-API-Key": "LznYKjBylk53uEV5UDlN57lolkR64tnr1VHwsM19",
            }

            # my_wallet
            my_address = 'QZ4JHEU6QEXZCB52W7ABKLOXNSH6PBOSFNU4VVJNYNCIRVAP6UWLB3IQMU'
            my_passphrase = 'cargo blush ocean cluster divert spider bunker gain excite shop jeans romance buzz loan potato stick people receive cross cheese unfair alter wild ability drop'
            
            # Client Wallet 
            client_pp = pp
            client_address = "Y33KRR6RH4AH5LV24TUCDI23L7TM3SN22LOJUPMYG7PFU4UHKEBQN5NCAY"
            print("Client address: {}".format(client_address))
            client_SK = mnemonic.to_private_key(client_pp)
            print("Client private key: {}".format(client_SK))
            CLIENT_PASSP = client_pp

            # Initialize an algod client
            algod_client = algod.AlgodClient(algod_token, algod_address, headers)

            # Get the relevant params from the algod    
            params = algod_client.suggested_params()
            params.flat_fee = True
            params.fee = 1000
            send_amount = 100000

            txn = transaction.PaymentTxn(sender=client_address, receiver=my_address, amt=send_amount, sp=params)
            signed_txn = txn.sign(client_SK)
            
            txid = algod_client.send_transaction(signed_txn)
            if(txid):
                print("Transaction sent, transaction ID: {}".format(txid))
                txnStr = str('Transaction Successful with TransactionID: {}'.format(txid))
                VERIFICATION_TOKEN = txid
            else:
                print("Transaction Failed")
                txnStr = str('Transaction Failed')
            return redirect(url_for('success', result=txnStr, VERIFICATION_TOKEN=VERIFICATION_TOKEN, BILL_VERIFICATION_TOKEN=BILL_VERIFICATION_TOKEN, hospitalName=hospitalName, review=review, client_pp=client_pp,rating=rating))
        
        else:
            return redirect(url_for('success', result='txnStr', VERIFICATION_TOKEN='VERIFICATION_TOKEN', BILL_VERIFICATION_TOKEN=BILL_VERIFICATION_TOKEN, hospitalName=hospitalName, review=review, client_pp=client_pp,rating=rating))
    return render_template('transaction.html')

#Successful transaction page
@app.route('/success', methods=['GET', 'POST'])
def success():
    
    conn = sqlite3.connect('hospitalReview.db')
    c = conn.cursor()
    VERIFICATION_TOKEN = request.args.get('VERIFICATION_TOKEN')
    hospitalName = request.args.get('hospitalName')
    CLIENT_PASSP = request.args.get('client_pp')
    review = request.args.get('review')
    rating = request.args.get('rating')
    BILL_VERIFICATION_TOKEN = request.args.get('BILL_VERIFICATION_TOKEN')
    if BILL_VERIFICATION_TOKEN=='Verified':
        if(VERIFICATION_TOKEN):
            #check genuinity of the review
            response = co.classify(inputs=[review], model='996de6b6-76c2-411b-97b7-aa58f15e75ad-ft')
            print(response)
            prediction=response.classifications[0].prediction
            confidence=response.classifications[0].confidence
            confidence=round(confidence*100,2)
            genuinity='genuine' if int(prediction)==0 else 'fake'
            #write the review to the database if it is genuine
            if genuinity=='genuine':
                #get id from hospital name
                # hospital=Hospital.query.filter_by(name=hospitalName).first()
                hospital=c.execute('SELECT * FROM hospital WHERE name=?',(hospitalName,)).fetchone()
                id=hospital[0]
                #add review to database
                numReview=c.execute('INSERT INTO review (name,review,hospital_id,date_created,confidence,rating) VALUES (?,?,?,?,?,?)',(CLIENT_PASSP,review,id,'05/10/22',confidence,rating))
                # newReview=Review(name='Anonymous',review=review,hospital_id=id,date_created='05/10/22',confidence=confidence,rating=rating)
                # DB.session.add(newReview)
                DB.session.commit()
                # Return the transaction back to client
                 # my_wallet
                algod_address = "https://testnet-algorand.api.purestake.io/ps2"
                algod_token = ""
                headers = {
                    "X-API-Key": "LznYKjBylk53uEV5UDlN57lolkR64tnr1VHwsM19",
                }

                # my_wallet
                my_address = 'QZ4JHEU6QEXZCB52W7ABKLOXNSH6PBOSFNU4VVJNYNCIRVAP6UWLB3IQMU'
                my_passphrase = 'cargo blush ocean cluster divert spider bunker gain excite shop jeans romance buzz loan potato stick people receive cross cheese unfair alter wild ability drop'
                
                client_pp = CLIENT_PASSP

                client_address = "Y33KRR6RH4AH5LV24TUCDI23L7TM3SN22LOJUPMYG7PFU4UHKEBQN5NCAY"
                print("Client address: {}".format(client_address))
                client_SK = mnemonic.to_private_key(client_pp)
                print("Client private key: {}".format(client_SK))

                # My wallet
                my_address = 'QZ4JHEU6QEXZCB52W7ABKLOXNSH6PBOSFNU4VVJNYNCIRVAP6UWLB3IQMU'
                my_sk = mnemonic.to_private_key(my_passphrase)
                print("My address: {}".format(my_address))
                print("My private key: {}".format(my_sk))


                # Initialize an algod client
                algod_client = algod.AlgodClient(algod_token, algod_address, headers)

                # Get the relevant params from the algod    
                params = algod_client.suggested_params()
                params.flat_fee = True
                params.fee = 1000
                send_amount = 110000

                # Send the transaction from my wallet to the client
                txn = transaction.PaymentTxn(my_address, params, client_address, send_amount)
                signed_txn = txn.sign(my_sk)


                txid = algod_client.send_transaction(signed_txn)
                
                return render_template('txnsucess.html', txnid=VERIFICATION_TOKEN,result=genuinity,confidence=confidence)
            else:
                return render_template('failure.html', txnid=VERIFICATION_TOKEN)

        else:
            return render_template('txnfailure.html')
    else:
        return render_template('billfailure.html', txnid=VERIFICATION_TOKEN)


if __name__ == '__main__':
    app.run(debug=True)

