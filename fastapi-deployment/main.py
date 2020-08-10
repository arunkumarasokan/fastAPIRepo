import uvicorn
from fastapi import FastAPI
import pickle


#openModel
pickle_in = open("ExxonOil.pickle", "rb")
linear = pickle.load(pickle_in)


app = FastAPI()

@app.get('/')

async def index():
    return{"text": "heloo there"}

@app.get('/items/{oilprice}')
async def get_items(oilprice):
    return {"oil price": oilprice}

#ML Aspect

@app.get('/predict/{oilprice}')
async def predict(oilprice):
    op = float(oilprice)
    prediction = linear.predict([[op]])
    predicted_value = prediction[0][0]
    return {"oil_price":oilprice, "EXXON Price" : predicted_value}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
