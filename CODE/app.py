from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
import pickle
from sklearn import preprocessing

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# SQLite DB setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Load ML model
with open('model_rf.pickle', 'rb') as f:
    model = pickle.load(f)

type_encoder = preprocessing.LabelEncoder()
type_encoder.fit(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"])

@app.route('/')
def home():
    if 'username' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            return 'Username already exists!'

        # hashed_password = generate_password_hash(password, method='sha256')
        # hashed_password = generate_password_hash(password, method='sha256')
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials!'
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        step = int(request.form['step'])
        type_val = request.form['type']
        amount = float(request.form['amount'])
        nameOrig = request.form['nameOrig']
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = request.form['nameDest']
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        type_encoded = type_encoder.transform([type_val])[0]
        nameOrig_encoded = hash(nameOrig) % (10 ** 8)
        nameDest_encoded = hash(nameDest) % (10 ** 8)

        isFlaggedFraud = 0  # or you can add this to the form if needed

        input_data = np.array([[step, type_encoded, amount,
                                nameOrig_encoded, oldbalanceOrg, newbalanceOrig,
                                nameDest_encoded, oldbalanceDest, newbalanceDest,
                                isFlaggedFraud]])

        prediction = model.predict(input_data)[0]
        result = "Fraudulent Transaction ❌" if prediction == 1 else "Legitimate Transaction ✅"
        return render_template('index.html', prediction=result)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=5000, debug=True)
