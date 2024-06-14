from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Load the tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
sum_model = AutoModelForSeq2SeqLM.from_pretrained("jwhong2006/wikisum")
fix_model = AutoModelForSeq2SeqLM.from_pretrained("jwhong2006/t5-PostOCRAutoCorrecttion")

def fix(text):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
    input_length = inputs.size(1)
    fix_ids = fix_model.generate(inputs, max_length=input_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    fixed = tokenizer.decode(fix_ids[0], skip_special_tokens=True)
    return fixed

@app.route('/summ', methods=['POST'])
def summ():
    data = request.get_json()
    text = data['text']
    print('\n\n')
    print(text)
    text = text.replace('- \n', '').replace('\n', ' ')
    print('\n\n')
    print(text)
    text = fix(text)
    print('\n\n')
    print(text)
    
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    input_length = inputs.size(1)

    # Generate the summary
    summary_ids = sum_model.generate(inputs, max_length=512, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return jsonify({'summary': summary})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)