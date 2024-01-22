# Digit classification with convolutional neural network CNN

Simple digit classification with convolutional neural network CNN.

## Setup
1. First, configure the environment variables:
```bash
export PIPENV_VENV_IN_PROJECT=1
export FLASK_APP=src
export FLASK_ENV=development
```

2. Then, install the dependencies:
```bash
pipenv install
```

3. Run the application:
```bash
pipenv run flask run --debug
```

## Production
Consider run this script
```bash
./start.sh
```

Then, run the app with:
```bash
# Run the app
pipenv run gunicorn -w 4 'src:app'
```
