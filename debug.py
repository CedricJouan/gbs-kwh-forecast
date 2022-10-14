
import sys,json, os
import pandas as pd

if __name__=="__main__":

	with open('./input_payloads/sample_single_payload_0.json', 'r') as f:
		data = json.load(f)

	df = pd.DataFrame(data['input_data'][0]['values'], columns = data['input_data'][0]['fields'])

	sys.path.append(os.path.join(os.getcwd(), "source"))
	os.chdir('./source')
	from master import main

	output = main(df)

	print(output)