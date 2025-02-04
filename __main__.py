
from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pymongo
from bson import ObjectId

from configuration.configuration import * 
#function that encode the textual arguemnt of the use and KB

def text_encoder(model, user_arg, KB):


	KB_enc=[]

	for i in range(len(KB)):

		KB_enc.append(model.text_embedding(KB[i]))

	user_arg_enc=model.text_embedding(user_argument)

	
	return KB_enc, user_arg_enc




#function that evaliate the similarity between a vector and a matrix of vector


def similarity_extraction(user_arg_enc, KB_enc):

	KB_sim=[]

	# to normilize the similarity I consider the value tha assume when the are two equal texts

	norm_term=np.dot(user_arg_enc,user_arg_enc)


	for i in range(len(KB_enc)):

		temp=[]

		for j in range(len(KB_enc[i])):

			#calcolo la similarity con il dot product normalizzandolo

			temp.append(np.dot(user_arg_enc,KB_enc[i][j])/norm_term)

		KB_sim.append(temp)	

	return KB_sim



def sort_matrix(KB, KB_sim):

	KB_sort=[]

	for i in range(len(KB_sim)):

		list_score, list_candidate = zip(*sorted(zip(KB_sim[i], KB[i]), reverse=True))

		KB_sort.append(list_candidate)

	return KB_sort



def max_method(KB_sort, KB_sim, id_list, thereshold):

	max_sim=0

	index_option=0

	for i in range(len(KB_sort)):

		if (max_sim <= KB_sim[i][0]):

			max_sim = KB_sim[i][0]

			index_option=i

	if max_sim < thereshold:

		return [False,id_list[index_option]]

	else:

		return [True,id_list[index_option]]



def argument_classification_ms(model, user_argument, KB, id_list, thereshold=0.8):


	KB_enc, user_arg_enc = text_encoder(model, user_argument, KB)


	KB_sim = similarity_extraction(user_arg_enc, KB_enc)


	KB_sort = sort_matrix(KB, KB_sim)


	return max_method(KB_sort, KB_sim, id_list, thereshold)


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




if __name__ == "__main__":


	print("user",USERNAME)


	print("password",PASSWORD)

	client = pymongo.MongoClient(DB_URL)  # Cambia l'URL se il server è remoto

	# Selezione del database e della collezione
	db = client[DB]
	collection = db[COL]

	# Lettura di tutti i documenti nella collezione
	documents = collection.find()

	# Iterazione e stampa di ogni documento
	print("Documenti trovati nella collezione:")

	KB=[]
	id_list=[]

	for i in documents:

		for j in i['arguments']:

			id_list.append(j['id'])
			KB.append(j['sentences'])

	'''import json

	#take in input the json file

	json_file=extract_json("dataset/KB.json")

	#convert the json file into a list of lists

	id_list, KB = from_json_to_ll(json_file)'''





	#calling the object BERT encoder

	BERT_Model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)

	#user argument (will be passed using paramethers)

	user_argument= "My name is"

	# thereshold of the semantic meaning

	thereshold=0.8

	print(argument_classification_ms(BERT_Model, user_argument, KB, id_list, thereshold))






















