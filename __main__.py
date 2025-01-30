
from model.BERTModel import BERT_Model
from sentence_transformers import SentenceTransformer, util
import numpy as np

#function that encode the textual arguemnt of the use and KB

def text_encoder(model, user_arg, KB):


	KB_enc=np.zeros([len(KB),len(KB[0])])

	for i in range(len(KB)):
		for j in range(len(KB[i])):

			KB_enc[i][j] = model.text_embedding(KB[i][j])

	user_arg_enc=model.text_embedding(user_argument)

	return KB_enc, user_arg_enc




#function that evaliate the similarity between a vector and a matrix of vector


def similarity_extraction(KB_enc, user_arg_enc):

	KB_sim=np.zeros([len(KB_enc),len(KB_enc[0])])

	for i in range(len(KB_enc)):
		for j in range(len(KB_enc[i])):

			KB_sim[i][j] = np.dot(KB_enc[i][j],user_arg_enc)

	return KB_sim



def sort_matrix(KB, KB_sim):

	KB_sort=[]

	for i in range(len(KB_sim)):

		list_score, list_candidate = zip(*sorted(zip(KB_sim[i], KB[i]), reverse=True))

		KB_sort.append(list_candidate)

	return KB_sort



def max_method(KB_sort, KB_sim, thereshold):

	max_sim=0

	index_option=0

	for i in range(len(KB_sort)):

		if (max_sim <= KB_sim[i][0]):

			max_sim = KB_sim[i][0]

			index=i

	if max_sim < thereshold:

		return index

	else:

		return "other"



def argument_classification_ms(model, user_argument, KB, thereshold=0.8):


	KB_enc, user_arg_enc = text_encoder(model, user_argument, KB)

	KB_sim = similarity_extraction(KB_enc, user_arg_enc)

	KB_sort = sort_matrix(KB, KB_sim)


	return max_method(KB_sort, KB_sim)





if __name__ == "__main__":


	BERT_Model = BERT_Model("multi-qa-mpnet-base-dot-v1", SentenceTransformer)

	user_argument= "Blablabla bullshit"


	KB=[["Ciao sono Marco", "Come stai?", "Io sto bene"],['Blablabla','Dog,dog,dog',"hhhhhhh"],["ppso","sjjsnd"]]

	print(argument_classification_ms(BERT_Model, user_argument, KB))






















