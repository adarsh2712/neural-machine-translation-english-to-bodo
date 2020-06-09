
import numpy as np
from flask import Flask, request, jsonify, Response
from flask import send_from_directory
import os
import json



app = Flask(__name__,static_url_path='')




@app.route('/api',methods=['POST','GET'])
def przedict():
    data = request.get_json(force=True)
    print(data)
    text=data['text']
    import nmt2 as pz
    js = json.dumps({'text':pz.ppredict(text)})    
    resp = Response(js, status=200, mimetype='application/json')
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    
    app.run(host="127.0.0.1", port=int("5000"), debug=True)
