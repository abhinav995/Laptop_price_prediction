import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load your model
with open(r"C:\Users\ABHINAV\OneDrive\Desktop\laptop prediction\laptop_prediction.pkl", 'rb') as f:
    load = pickle.load(f)

# Load your dataframe
df = pd.read_csv("/Users/ABHINAV/OneDrive/Desktop/laptop prediction/cleaned_laptop_data.csv")

@app.route('/')
def home():
    Company = sorted(df["Company"].unique())
    TypeName = sorted(df["TypeName"].unique())
    Inches = sorted(df["Inches"].unique())
    Cpu = sorted(df["Cpu"].unique())
    Ram = sorted(df["Ram"].unique())
    Memory1 = sorted(df["Memory1"].unique())
    Memory2 = sorted(df["Memory2"].unique())
    Gpu = sorted(df["Gpu"].unique())
    OpSys = sorted(df["OpSys"].unique())
    return render_template('laptop.html', Company=Company, TypeName=TypeName, Inches=Inches, Cpu=Cpu, Ram=Ram, Memory1=Memory1, Memory2=Memory2, Gpu=Gpu, OpSys=OpSys)

@app.route('/predict', methods=['POST'])
def pred():
    if request.method == 'POST':
        company = request.form['company']
        typename = request.form['typename']
        inches = request.form['inches']
        cpu = request.form['cpu']
        ram = request.form['ram']
        memory1 = request.form['memory1']
        memory2 = request.form['memory2']
        gpu = request.form['gpu']
        opsys = request.form['opsys']
        
        # Create input dataframe for prediction
        input_data = pd.DataFrame([[company, typename, inches, cpu, ram, memory1, memory2, gpu, opsys]], 
                                  columns=['Company', 'TypeName', 'Inches', 'Cpu', 'Ram', 'Memory1', 'Memory2', 'Gpu', 'OpSys'])
        
        # Make prediction using the loaded model
        prediction = load.predict(input_data)[0]
        
        return render_template('laptop.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
