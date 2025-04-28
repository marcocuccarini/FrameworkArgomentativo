
from model.T5Model import T5_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pymongo
from bson import ObjectId
import random
from configuration.configuration import * 
import torch



#function that evaliate the similarity between a vector and a matrix of vector


def json_extract(documents):
	KB=[]
	id_list=[]
	KB_temp=[]
	id_list_temp=[]

	for i in documents:




		for j in i['arguments']:

			id_list.append(j['id'])
			KB.append(j['sentences'])

		KB_temp.append(KB)
		id_list_temp.append(id_list)

	return id_list_temp, KB_temp


def sentence_augmentation(T5_Model, sentences, k, repetition_penalty=15.0, diversity_penalty=1.0):

	r_num=random.randrange(len(sentences))

	while (len(sentences)<k):


		sentences.append(T5_Model.generate_title(sentences[r_num], repetition_penalty, diversity_penalty)[0])


	return sentences


if __name__ == "__main__":


	print("user",USERNAME)


	print("password",PASSWORD)

	client = pymongo.MongoClient(DB_URL)  # Cambia l'URL se il server è remoto

	# Selezione del database e della collezione
	db = client[DB]
	collection = db[COL]

	# Lettura di tutti i documenti nella collezione
	documents = collection.find()


	#convert the json file into a list of lists

	T5_Model= T5_Model(MODEL, TOKENIZER, TOKEN)

	id_list, KB = json_extract(documents)


	print(KB)
	print(id_list)


	for i in range(len(KB)):

		KB[i]=sentence_augmentation(T5_Model, KB[0][i], k, repetition_penalty, diversity_penalty)





	print(KB)
	print(id_list)

































