import tensorflow as tf
import numpy as np
from flask import Flask, request

# load_inference = load.signatures["serving_default"]  # 이게 무슨뜻?

app = Flask(__name__)
new_model = None

if __name__ == '__main__':
    new_model = tf.keras.models.load_model('def_model/classifier')

    app.run(host='0.0.0.0', port=2431)


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    result = new_model
    return "text"
    # result = load_inference(tf.constant(data['images'], dtype=tf.float32) / 255.0)
    # return str(np.argmax(result['dense_41'].numpy()))
