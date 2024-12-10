from openai import OpenAI
import json
import os
import pandas as pd
from pprint import pprint
import datetime

gik_open_ai = os.getenv("GIK_OPEN_AI")
client = OpenAI(api_key=gik_open_ai)

