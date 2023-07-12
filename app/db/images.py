import mongoengine as db


class ImagesDBModel(db.Document):
    image_id = db.StringField(primary_key=True)
    image_name = db.StringField(required=True)
    image_size = db.IntField(required=True)
    image = db.ImageField(required=True)
    result = db.StringField(required=False)