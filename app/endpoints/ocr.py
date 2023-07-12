from fastapi import UploadFile, APIRouter
from PIL import Image
import io
import numpy as np
from OCR.utils import predict_image
from app.config import model
from app.config import mongodb
from app.db.connection import Connection
from app.db.images import ImagesDBModel
from loguru import logger
import uuid


router = APIRouter()

@router.post('/predict')
async def predict(file: UploadFile):
    try:
        with Connection(db=mongodb.mongo_db, host=mongodb.mongo_host, port=mongodb.mongo_port) as conn:
            byte_image = io.BytesIO(await file.read())
            image = Image.open(byte_image)
            image_array = np.array(image)

            prediction, _ = predict_image(image_array, model)

            img_db = ImagesDBModel(image_id=str(uuid.uuid4()),
                                   image_name=file.filename,
                                   image_size=file.size,
                                   image=byte_image,
                                   result=prediction)

            img_db.save()
            if prediction is None:
                return {'prediction': "None"}

            return {'prediction': prediction[0]}
    except Exception as e:
        logger.exception(f"Error predicting the image: {str(e)}")
        return f"Error predicting the image.Please contact support!"

