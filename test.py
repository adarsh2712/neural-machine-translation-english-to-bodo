import pickle

with open('english-bodo-test.pkl', 'rb') as f:
	data = pickle.load(f)
	
print(data)