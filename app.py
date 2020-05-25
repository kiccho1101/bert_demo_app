import flask
from flask import Flask, request, render_template
import json
import main
from typing import List

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/vectorize_sentences", methods={"post"})
def vectorize_sentences():
    try:
        input_sentences: List[str] = request.json["input_text"].split("\n")
        res = main.vectorize_sentences(input_sentences)
        return app.response_class(
            response=json.dumps(res), status=200, mimetype="application/json"
        )
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(
            response=json.dumps(err), status=500, mimetype="application/json"
        )


@app.route("/umap_comp", methods={"post"})
def umap_comp():
    try:
        data: List[List[float]] = request.json["data"]
        n_neighbors = int(request.json["n_neighbors"])
        min_dist = float(request.json["min_dist"])
        res = main.umap_comp(data=data, n_neighbors=n_neighbors, min_dist=min_dist)
        return app.response_class(
            response=json.dumps(res), status=200, mimetype="application/json"
        )
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(
            response=json.dumps(err), status=500, mimetype="application/json"
        )


@app.route("/sentiment_analysis", methods={"post"})
def sentiment_analysis():
    try:
        input_sentence: str = request.json["input_text"]
        res = main.sentiment_analysis(input_sentence)
        return app.response_class(
            response=json.dumps(res), status=200, mimetype="application/json"
        )
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(
            response=json.dumps(err), status=500, mimetype="application/json"
        )


@app.route("/get_end_predictions", methods=["post"])
def get_prediction_eos():
    try:
        input_text = " ".join(request.json["input_text"].split())
        input_text += " <mask>"
        top_k = request.json["top_k"]
        res = main.get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(
            response=json.dumps(res), status=200, mimetype="application/json"
        )
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(
            response=json.dumps(err), status=500, mimetype="application/json"
        )


@app.route("/get_mask_predictions", methods=["post"])
def get_prediction_mask():
    try:
        input_text = " ".join(request.json["input_text"].split())
        top_k = request.json["top_k"]
        res = main.get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(
            response=json.dumps(res), status=200, mimetype="application/json"
        )
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(
            response=json.dumps(err), status=500, mimetype="application/json"
        )


if __name__ == "__main__":
    print("starting server on http://loaclhost:8000")
    app.run(host="0.0.0.0", debug=True, port=8000, use_reloader=False)
