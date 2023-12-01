from flask_frozen import Freezer
from openaid import app

freezer = Freezer(app)


@freezer.register_generator
def show_schwerpunkte():
    yield '/sectors/all/'

if __name__ == '__main__':
    freezer.freeze()
