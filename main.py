from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel

model = joblib.load('model (1).pkl')
scaler = joblib.load('scaler (2).pkl')

avocado_app = FastAPI()

class AvocadoSchema(BaseModel):
    firmness: float
    hue: int
    saturation: int
    brightness: int
    sound_db: int
    weight_g: int
    size_cm3: int
    color_category: str

@avocado_app.post('/predict/', response_model=dict)
async def predict_avocado(avocado: AvocadoSchema):
    avocado_dict = avocado.dict()

    color_category = avocado_dict.pop('color_category')
    color_category1_0 = [
        1 if color_category == 'black' else 0,
        1 if color_category == 'dark green' else 0,
        1 if color_category == 'green' else 0,
        1 if color_category == 'purple' else 0,
    ]

    avocado_data = list(avocado_dict.values()) + color_category1_0

    scaled_data = scaler.transform([avocado_data])
    pred = model.predict(scaled_data)[0]
    labels = {
        0: 'hard',
        1: 'pre-conditioned',
        2: 'breaking',
        3: 'firm-ripe',
        4: 'ripe',
    }
    pred_label = labels[int(pred)]
    return {'predict': pred_label}


if __name__ == '__main__':
    uvicorn.run(avocado_app, host='127.0.0.1', port=8001)



