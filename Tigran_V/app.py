from distutils.log import debug
from operator import methodcaller
from flask import Flask, request, render_template, url_for, redirect


app = Flask(__name__)

@app.route("/") #. Home Page API
def hellow_world():
  '''
  Creating Home page.
  This will have information about ...
  '''
  return render_template('home.html')




@app.route("/about")
def about():
  '''
  Creating About page.
  This will have information about ...
  '''
  return render_template('about.html')


@app.route("/contact")
def contact_page():
  '''
  Creating contact page.
  This will have information about ...
  '''
  return render_template('contact.html')

@app.route("/howto")
def how_to():
  '''
  Creating how-to page.
  This will have information about ...
  '''
  return render_template('howto.html')



@app.route("/predict")

def prediction():
  '''
  Creating Home page.
  This will have information about ...
  '''
  return None





#checking if code is run on local environment
if __name__ == "__main__":
  app.run(debug = True) # debug =True makes it easy to debug as we  make changes

