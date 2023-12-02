import cohere 
co = cohere.Client('INSERT API KEY HERE') # Insert API Key Here

from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

def result_gen(phrase_input):
  # Processes the input to decide toxicity level
  response = co.classify( 
    model='cohere-toxicity', 
    inputs=[phrase_input]) 

  # Grabs results from API call
  result = format(response.classifications)
  split1 = result.split('\n')
  sliced = split1[2:6]
  pre_result = sliced[0]
  pre_confidence = sliced[3]
  mid_result = pre_result.split(': ')
  mid_confidence = pre_confidence.split(': ')
  final_result = mid_result [1]
  final_confidence = mid_confidence [1] 
  final_confidence = round(((float(final_confidence)) * 100), 2) 

  # Calls decider function
  return final_result, final_confidence

# Renders homepage
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def post():
    temp = format(request.form)
    temp2 = temp.split('\'')
    a, b = result_gen(temp2[3])

    print (a, b)

    if a == 'TOXIC':
      confidence_level = 100 - b
      if confidence_level > 80:
        print ('This interaction is Toxic.')
        print (f'{confidence_level}%')
        return render_template('toxic.html')
      else:
        print ('This interaction is somewhat Toxic.')
        print (f'{confidence_level}%')
        return render_template('likelytoxic.html')
    else:
      confidence_level = b
      if confidence_level > 70:
        print ('This interaction is Not Toxic.')
        print (f'{confidence_level}%')
        return render_template('likelynot.html')
      else:
        print ('This interaction is likely Not Toxic.')
        print (f'{confidence_level}%')
        return render_template('nottoxic.html')

if __name__ == "__main__":
    app.run(debug=True)