import sys
from flask import Flask

print(sys.path)
app = Flask(__name__)

print(app)

import src.predictionServer