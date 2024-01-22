export PIPENV_VENV_IN_PROJECT=1
export FLASK_APP=src
export FLASK_ENV=production

pipenv install
pipenv install gunicorn

# Run the app
# pipenv run gunicorn -w 4 'src:app'
