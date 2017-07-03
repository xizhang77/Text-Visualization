# -*- coding: utf-8 -*-]
from flask import Flask

app = Flask(__name__)

import application.views
import application.cluster