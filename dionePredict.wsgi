python_home = '/home/thanasis/PycharmProjects/dionePredict/venv'


activate_this = python_home + '/bin/activate_this.py'
execfile(activate_this, dict(__file__=activate_this))

from dionePredict import app as application
application.secret_key = 'thanasis'
