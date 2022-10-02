from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from MushroomML import Model

# Create a Flask instance
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # attributes
    cap_shape = request.args.get('cap_shape')
    cap_surface = request.args.get('cap_surface')
    cap_color = request.args.get('cap_color')
    bruises = request.args.get('bruises')
    odor = request.args.get('odor')
    gill_attach = request.args.get('gill_attach')
    gill_space = request.args.get('gill_space')
    gill_size = request.args.get('gill_size')
    gill_color = request.args.get('gill_color')
    stalk_shape = request.args.get('stalk_shape')
    stalk_root = request.args.get('stalk_root')
    stalk_surface_above = request.args.get('stalk_surface_above')
    stalk_surface_below = request.args.get('stalk_surface_below')
    stalk_color_above = request.args.get('stalk_color_above')
    stalk_color_below = request.args.get('stalk_color_below')
    veil_color = request.args.get('veil_color')
    ring_number = request.args.get('ring_number')
    ring_type = request.args.get('ring_type')
    spore_print_color = request.args.get('spore_print_color')
    population = request.args.get('population')
    habitat = request.args.get('habitat')
    
    
    data = {'cap-shape': cap_shape, 'cap-surface': cap_surface, 
            'cap-color': cap_color, 'bruises': bruises, 'odor': odor,
            'gill-attachment': gill_attach, 'gill-spacing': gill_space,
            'gill-size': gill_size, 'gill-color': gill_color,
            'stalk-shape': stalk_shape, 'stalk-root': stalk_root,
            'stalk-surface-above-ring': stalk_surface_above,
            'stalk-surface-below-ring': stalk_surface_below,
            'stalk-color-above-ring': stalk_color_above,
            'stalk-color-below-ring': stalk_color_below,
            'veil-color': veil_color, 'ring-number': ring_number,
            'ring-type': ring_type, 'spore-print-color': spore_print_color,
            'population': population, 'habitat': habitat}
    
    # get prediction
    labelencoder=LabelEncoder()
    data = labelencoder.fit_transform(data)
    
    # use trained decision tree
    model = Model.__init__()
    prediction = model.predict(data)
    
    # return prediction
    return prediction
