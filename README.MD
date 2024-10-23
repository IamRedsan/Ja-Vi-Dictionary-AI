**Create an environment**
`python -m venv .venv`

**Activate the environment**
`.venv\Scripts\activate`

> if error, run this first: `Set-ExecutionPolicy Unrestrict -Scope Process`

**Install libraries**
`pip install -r .\requirements.txt`

**Donwnload models**
Download my_model_v1_12_weights.h5 and my_model_v3_23_weights.h5 and put into .\modelWeights folder

**Donwnload VnCoreNLP**
`git clone https://github.com/vncorenlp/VnCoreNLP.git`

**Update path**
Change ABSOLUTE path in ./utils/model.py, line 30, 354 and 372

**Run flask**
`flask --app flaskr run --host=0.0.0.0`
