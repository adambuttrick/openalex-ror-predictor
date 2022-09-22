import requests


def query(affilitation_strig):
	url = 'http://127.0.0.1:5000/invocations'
	test_data ={'affiliation_string':[affilitation_strig]}
	response = requests.post(url, json = test_data)
	return response

if __name__ == '__main__':
	# Institution ID and ROR ID
	print(query('School of Engineering, Tampere University, Finalnd').text)
	# Institution ID, but no ROR ID
	print(query('Department of Probation, New York, USA').text)
	# Neither institution ID, nor ROR ID
	print(query('An organization that does not exist').text)