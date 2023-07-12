import tensorflow as tf
from pydantic import BaseModel


class MongoDB(BaseModel):
    mongo_db: str = "ocr"
    mongo_host: str = "127.0.0.1"
    mongo_port: str = 27017

mongodb = MongoDB()

model = tf.keras.models.load_model("../OCR/models_saved/v200bs1024/ocr_model_200e.hdf5")
