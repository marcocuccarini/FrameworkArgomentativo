
from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json

from configuration.configuration import * 
#function that encode the textual arguemnt of the use and KB

def encode_json(model, json_file):

	for i in range(len(json_file)):
		for j in range(len(json_file[i]['arguments'])):

			json_file[i]['arguments'][j]['sentences'] = model.text_embedding(json_file[i]['arguments'][j]['sentences'])

	return json_file




def extract_json(path):

	 with open(path) as f:

	 	d = json.load(f)
	 	return d



if __name__ == "__main__":

	# Lettura di tutti i documenti nella collezi

	json_file = extract_json("dataset/KB1.json")


	BERT_Model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)


	json_file_encoded = encode_json(BERT_Model, json_file)

	with open("data.json", "w") as json_file:
		json.dump(data, json_file_encoded, indent=4)







	#convert the json file into a list of lists



























