from flask import Flask, render_template, request, redirect, url_for, send_file
from werkzeug.utils import secure_filename
import pandas as pd
#You need to use following line [app Flask(__name__]

UPLOAD_FOLDER = './static/uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import sqlite3, os
from sqlite3 import Error
from attribute import rarity_score_calculation
import sys

def getTokens(collectionName):
  con = sqlite3.connect("collections.sqlite")
  cursor = con.cursor()
  command = "SELECT * FROM " + collectionName
  cursor.execute(command)
  results = cursor.fetchall()
  con.close()
  return results

def getTableNames():
  con = sqlite3.connect("collections.sqlite")
  cursor = con.cursor()
  cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
  results = cursor.fetchall()
  return results

def getColumnNames(collectionName):
  con = sqlite3.connect("collections.sqlite")
  cursor = con.cursor()
  command = "SELECT * FROM " + collectionName
  cursor.execute(command)
  names = [description[0] for description in cursor.description]
  return names

def requestHandlerFile(request, nameOfFile):
    f = request.files[nameOfFile]
    f.save(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], f.filename))
    return data



@app.route("/add")
def add():
  return render_template("add.html")


@app.route("/adding", methods=['POST'])
def adding():
  if request.method == 'POST':
    data_transactions = requestHandlerFile(request, "transactions")
    data_attributes = requestHandlerFile(request, "attributes")
    my_dfs = rarity_score_calculation(data_attributes, data_transactions)

    with sqlite3.connect("collections.sqlite") as my_db:
      for table_name, df in my_dfs.items():
          df.to_sql(table_name, my_db, if_exists="replace")
      my_db.commit()
    return redirect("index.html")


@app.route("/")
def index():
  collectionNames = getTableNames()
  for i in range(len(collectionNames)):
    collectionNames[i] = collectionNames[i][0]

  # (C2) RENDER HTML PAGE
  return render_template("index.html", usr=collectionNames)


@app.route('/collection', methods = ['POST'])
def result():
    if request.method == 'POST':
      print(request.form['submit_button'], file=sys.stderr)
      if request.form['submit_button'] == 'Add new collection':
        return render_template("add.html")
      tokens = getTokens(request.form['submit_button'])
      columnNames = getColumnNames(request.form['submit_button'])
      collectionName = request.form['submit_button'].replace('_', '')

      for i in range(0, len(tokens)):
        tokens[i] = list(tokens[i])
        tokens[i][-1] = tokens[i][-1].replace(' ', '')
        tokens[i][-1] = tokens[i][-1].replace(collectionName, '')
        tokens[i] = tuple(tokens[i])

      return render_template("result.html", usr=tokens, columns=columnNames, collectionName=request.form['submit_button'])

if __name__ == '__main__':
   app.run(debug = True)

