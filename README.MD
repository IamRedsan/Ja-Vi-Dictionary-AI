**Create an environment**
> Windows: `py -3 -m venv .venv`

> Linux: `python3 -m venv .venv`

**Activate the environment**
> Windows: `.venv\Scripts\activate`

> Linux: `. .venv/bin/activate`

> if error on Windows, run this first: `Set-ExecutionPolicy Unrestrict -Scope Process`

**Install libraries**
> `pip install -r requirements.txt`

**Model weights**
> flaskr\utils\model: [2 files](https://drive.google.com/drive/folders/1X894dIJKhR5GBxExvRb2HvVUlkAy4AYm?usp=drive_link)

**Run flask**
> `flask --app flaskr run --host=0.0.0.0`

**Port Forwarding WSL 2 to Your LAN**
> `netsh interface portproxy add v4tov4 listenport=[PORT] listenaddress=0.0.0.0 connectport=[PORT] connectaddress=[WSL_IP]`

> You can view all forwarded ports with `netsh interface portproxy show v4tov4`
