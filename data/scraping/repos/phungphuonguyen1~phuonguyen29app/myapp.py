import streamlit as st
from langchain.llms import OpenAI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
import joblib

st.write("""
# Simple prediction

This app preidicts systems' features
""")

st.sidebar.header("User input parameters")
#materiallist=['Au','Ag']

#'Po_NP', 'NP_NP', 'D_aim', 'Phi', 'Chain length', 'distance', 'function'
def user_input_features():
    ponp=st.sidebar.slider('Interaction between polymers and nanoparticles: ',0.0,2.5, 0.4)
    npnp=st.sidebar.slider('Interaction between nanoparticles and nanoparticles: ',0.0,2.5, 0.4)
    d=st.sidebar.slider('Diameter of nanoparticles: ',1,10,4)
    phi=st.sidebar.slider('Number of particles: ',0.001,0.01,0.002)
    cLength=st.sidebar.slider('Length of polymer chain: ',5,100,20)
    #mats= st.sidebar.selectbox('Material',materiallist)
    #radi=st.sidebar.slider('Radius',20,100,30)
    st.sidebar.subheader('Distance Range')
    distance_str_min = st.sidebar.text_input('Minimum distance in nm: ','0.075')
    distance_str_width_range = st.sidebar.text_input('Width range in nm: ','150')
    distance_min= float(distance_str_min)
    distance_range= float(distance_str_width_range)
    #if mats=='Au':
        #mate=1
    #if mats=='Ag':
        #mate=2
    Po_NP=pd.DataFrame({'Po_NP':[ponp]*2000})
    NP_NP=pd.DataFrame({'NP_NP':[npnp]*2000})
    D_aim=pd.DataFrame({'D_aim':[d]*2000})
    Phi=pd.DataFrame({'Phi':[phi]*2000})
    Chain_length=pd.DataFrame({'Chain_length':[cLength]*2000})
    distance = pd.DataFrame({'distance': np.linspace(distance_min, distance_min + distance_range, 2000)})
    features=pd.concat([Po_NP,NP_NP,D_aim,Phi,Chain_length,distance], axis=1)
    features.columns = ['Po_NP','NP_NP','D_aim','Phi','Chain length','distance']
    #features=data
    return features
df = user_input_features()
#st.write(df)
st.subheader('User input parameter')
st.write("""
        Interaction between polymers and nanoparticles: {},
        Interaction between nanoparticles and nanoparticles: {},
        Diameter of nanoparticles: {},
        Number of particle: {},
        Length of polymer chain: {},
        Distance range: {} - {} nm
         """.format(max(df['Po_NP']),max(df['NP_NP']),max(df['D_aim']),max(df['Phi']),max(df['Chain length']),min(df['distance']),max(df['distance'])))
# Load the model
model = joblib.load('model.pkl')

# Print the type and structure of the loaded object
st.write(type(model))
st.write(model)
st.write(df)
predictions1=model.predict(df)
st.write('max predicts: ',max(predictions1))
st.write('min predicts: ',min(predictions1))
st.subheader('Prediction')

fig, ax = plt.subplots()
ax.scatter(df['distance'],predictions1)
ax.set_xlabel('distance')
ax.set_ylabel('density')
ax.set_title('Prediction')

# Display the plot in Streamlit
st.pyplot(fig)

# -- Allow data download
download = df
df = pd.DataFrame(download)
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
fn =  str(max(df['Po_NP']))+' - ' +str(max(df['NP_NP']))+str(max(df['D_aim']))+str(max(df['Phi']))+str(max(df['Chain length']))+' - '+str(min(df['distance']))+'/'+str(max(df['distance'])) + '.csv'
href = f'<a href="data:file/csv;base64,{b64}" download="{fn}">Download Data as CSV File</a>'
st.markdown(href, unsafe_allow_html=True)
