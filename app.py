from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from MushroomML import Model
import random 

# Create a Flask instance
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # randomized attributes 
    gill_attachmentR = ['a', 'd', 'f','n']
    gill_spacingR = ['c', 'w', 'd']
    gill_sizeR = ['b', 'n']
    stalk_shapeR = ['e', 't']
    stalk_rootR = ['b', 'c', 'u', 'e', 'z', 'r']
    stalk_surfaceR = ['f','y','k','s'] #use for above and below
    stalk_colorR = ['n','b','c','g','o','p','e','w','y'] #use for above and below
    veil_colorR = ['n','o','w','y']
    ring_numberR = ['n' 'o', 't']
    ring_typeR = ['c','e','f','l','n','p','s','z']
    spore_print_colorR = ['k','n','b','h','r','o','u','w','y']
    habitatR = ['g','l','m','p','u','w','d']

    # attributes
    cap_shape = request.args.get('cap_shape')
    cap_surface = request.args.get('cap_surface')
    cap_color = request.args.get('cap_color')
    bruises = request.args.get('bruises')
    odor = request.args.get('odor')
    gill_attach = random.choice(gill_attachmentR) #Randomized 
    gill_space = random.choice(gill_spacingR) #Randomized 
    gill_size = random.choice(gill_sizeR) #Randomized 
    gill_color = request.args.get('gill_color')
    stalk_shape = random.choice(stalk_shapeR) #Randomized 
    stalk_root = random.choice(stalk_rootR) #Randomized 
    stalk_surface_above = random.choice(stalk_surfaceR) #Randomized 
    stalk_surface_below = random.choice(stalk_surfaceR) #Randomized 
    stalk_color_above = random.choice(stalk_colorR) #Randomized 
    stalk_color_below = random.choice(stalk_colorR) #Randomized 
    veil_color = random.choice(veil_colorR) #Randomized 
    ring_number = random.choice(ring_numberR) #Randomized 
    ring_type = random.choice(ring_typeR) #Randomized 
    spore_print_color = random.choice(spore_print_colorR) #Randomized 
    population = request.args.get('population')
    habitat = random.choice(habitatR) #Randomized 
    
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
    
    data_ar = []
    
    # get prediction
    for key, value in data.items():
        data_ar.append(value)

    # use trained decision tree
    model = Model()
    prediction = model.getStatus(data_ar)
    
    # return prediction
    return prediction
