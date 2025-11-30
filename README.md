## create elastic search container
```sh
pip install elasticsearch==8.13.0

docker compose up -d
```

## requirements.txt
```sh
pip freeze > requirements.txt

pip install -r requirements.txt
```

## run app
```sh
python app.py
```