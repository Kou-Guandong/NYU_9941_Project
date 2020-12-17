from flask import Flask, request, jsonify, render_template
from utils.utils import *


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def results():

    data = request.form
    job_title = data['title']
    job_desc = data['desc']

    result = predict(job_title, job_desc)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)  # for development
    # app.run(debug=False)  # for production
