
from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np

from configuration.configuration import * 
#function that encode the textual arguemnt of the use and KB

def text_encoder(model, user_arg, KB):


	KB_enc=[]

	for i in range(len(KB)):

		KB_enc.append(model.text_embedding(KB[i]))

	user_arg_enc=model.text_embedding(user_argument)

	
	return KB_enc, user_arg_enc




	


def extract_json(path):

	 with open(path) as f:

	 	d = json.load(f)
	 	return d

def from_json_to_ll(json_file):

	KB=[]

	id_list=[]

	for i in json_file:

		temp=[]

		id_list.append(i['id'])
		
		for j in i['phrases']:

			temp.append(j)

		KB.append(temp)

	return id_list, KB

def from_json_to_ll2(documents):
	KB=[]
	id_list=[]

	for i in documents:

		for j in i['arguments']:

			id_list.append(j['id'])
			KB.append(j['sentences'])

	return id_list, KB


if __name__ == "__main__":

	# Lettura di tutti i documenti nella collezi

	json_file=extract_json("dataset/KB.json")


	id_list, KB = from_json_to_ll2(documents)



	BERT_Model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)


	#convert the json file into a list of lists

	KB_enc, user_arg_enc = text_encoder(BERT_Model, KB)


























