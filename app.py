from flask import Flask, render_template, request

# Create a Flask instance
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    
    person = request.args.get('person')
    color = request.args.get('color')
    # add attribute here
    
    data = {"hello": person, "color": color, "foo": "bar"}
    return data

