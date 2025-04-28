
from model.T5Model import T5_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np
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

def extract_json(path):
    """Extracts JSON data from a given file path."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)



if __name__ == "__main__":

	import json

	# Load the JSON file
	arg_graph = extract_json("arg_dic.json")

	# Initialize the T5 model
	t5_model = T5_Model(MODEL, TOKENIZER, TOKEN)

	# Iterate over the argument graph
	for row in arg_graph:
	    for argument in row['arguments']:
	        # Perform sentence augmentation
	        argument['sentences'] = sentence_augmentation(
	            t5_model, 
	            argument['sentences'], 
	            k, 
	            repetition_penalty, 
	            diversity_penalty
	        )

	# Save the updated arg_graph to a new JSON file
	with open('arg_dic_supp_augmented.json', 'w', encoding='utf-8') as f:
	    json.dump(arg_graph, f, ensure_ascii=False, indent=4)



































