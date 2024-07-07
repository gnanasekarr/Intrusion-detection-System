import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(r'E:\Gnanasekar\Project\intrusion detection system\Deployment-flask-master\model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    proto_types={"tcp":0.0,"icmp":1.0,"udp":2.0}
    protocol_changing= int_features[0]
    ser1=['http', 'telnet', 'ftp_data', 'ftp', 'login', 'imap4', 'private',
       'smtp', 'time', 'mtp', 'rje', 'link', 'remote_job', 'gopher',
       'ssh', 'name', 'finger', 'whois', 'domain', 'eco_i', 'nnsp',
       'http_443', 'exec', 'shell', 'printer', 'efs', 'courier', 'uucp',
       'klogin', 'kshell', 'ctf', 'pop_3', 'nntp', 'ecr_i', 'tim_i',
       'daytime', 'systat', 'hostnames', 'sunrpc', 'other', 'netstat',
       'supdup', 'csnet_ns', 'vmnet', 'uucp_path', 'netbios_ns',
       'sql_net', 'Z39_50', 'domain_u', 'pm_dump', 'IRC', 'auth', 'ntp_u']
    ser2=[18, 46, 15, 14, 24, 20, 36, 40, 48, 25, 38, 23, 37, 16, 42, 26, 13,
       52,  7,  9, 29, 19, 12, 39, 35, 11,  3, 49, 21, 22,  5, 34, 30, 10,
       47,  6, 45, 17, 43, 32, 28, 44,  4, 51, 50, 27, 41,  1,  8, 33,  0,
        2, 31]
    serv_dict = dict(zip(ser1, ser2))
    service_changing = int_features[1]
    int_features[0]=proto_types[protocol_changing]
    int_features[1]=serv_dict[service_changing]
    final_features = [np.array(int_features)]
    
    prediction = model.predict(final_features)
 
    n=int(prediction)
    attack=['imap', 'ipsweep', 'smurf', 'back', 'nmap', 'neptune', 'teardrop', 'satan',
    'spy', 'normal', 'buffer_overflow', 'multihop', 'phf', 'portsweep', 'loadmodule', 'rootkit', 
    'guess_passwd', 'land', 'pod', 'warezmaster', 'class', 'warezclient', 'perl','ftp_write']
    

    return render_template('index.html', prediction_text=' The type of attack which is performed {}'.format(attack[n]))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)