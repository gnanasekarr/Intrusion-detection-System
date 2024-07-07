import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'protocol_type':2, 'service':9, 'src_bytes':6, 'dst_bytes':2, 'logged_in':9,
                            'count':2,'srv_count':20,'srv_diff_host_rate':2,'dst_host_count':2,'dst_host_srv_count':2,
                            'dst_host_same_src_port_rate':2,'dst_host_srv_diff_host_rate':2,'dst_host_same_srv_port_rate':2,
                            'dst_host_same_srv_port_rate':2})

print(r.json())