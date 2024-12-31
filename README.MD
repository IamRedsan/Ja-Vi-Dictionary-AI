**Create an environment**
`python -m venv .venv`

**Activate the environment**
`.venv\Scripts\activate`

> if error, run this first: `Set-ExecutionPolicy Unrestrict -Scope Process`

**Install libraries**
`pip install -r .\requirements.txt`

**Run flask**
`flask --app flaskr run --host=0.0.0.0`
