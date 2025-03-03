from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask_cors import CORS  # Import CORS from flask_cors

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("fine_tuned_gpt2_qa")

# Function to generate an answer given a question
def generate_answer(question, max_length=50):
    input_text = f"Question: {question} Answer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate answer
    output = model.generate(input_ids, max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove the question part from the answer
    answer = answer.split("Answer:")[1].strip()

    return answer

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route to handle form submission
@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        # Get the input text from the form
        user_input = request.form['user_input']
        
        # Process the input using the model
        predicted_answer = generate_answer(user_input)
        
        # Return the processed output to the HTML template
        return render_template('index.html', user_input=user_input, predicted_answer=predicted_answer)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
