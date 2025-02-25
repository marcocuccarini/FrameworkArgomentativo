
from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pymongo
from bson import ObjectId

from configuration.configuration import * 



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



def argument_classification_ms(model, user_argument, KB, KB_enc, id_list, thereshold=0.8):

	

	user_arg_enc=model.text_embedding(user_argument)


	KB_sim = similarity_extraction(user_arg_enc, KB_enc)


	KB_sort = sort_matrix(KB, KB_sim)


	return max_method(KB_sort, KB_sim, id_list, thereshold)


def extract_json(path):

	 with open(path) as f:

	 	d = json.load(f)
	 	return d



def json_extract(documents):
	KB=[]
	id_list=[]

	for i in documents:

		for j in i['arguments']:

			id_list.append(j['id'])
			KB.append(j['sentences'])

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

	collection_enc = db[COL_ENC]

	# Iterazione e stampa di ogni documento

	documents_enc = collection_enc.find()


	'''import json

	#take in input the json file

	json_file=extract_json("dataset/KB.json")'''



	#convert the json file into a list of lists

	id_list, KB = json_extract(documents)

	id_list, KB_enc = json_extract(documents_enc)


	#calling the object BERT encoder

	BERT_Model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)

	#user argument (will be passed using paramethers)

	user_argument= "My name is"

	# thereshold of the semantic meaning

	thereshold=0.8

	print(argument_classification_ms(BERT_Model, user_argument, KB, KB_enc, id_list, thereshold))






















