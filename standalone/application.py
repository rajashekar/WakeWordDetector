from flask import Flask, render_template

application = app = Flask(__name__)


@app.route("/")
def index():
    """Return the client application."""
    return render_template("audio/main.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0')