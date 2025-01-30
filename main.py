


def text_encoder(model, user_arg, KB):


	KB_enc=np.array[len(KB),len(KB[i])]

	for i in range(len(KB)):
		for j in range(len(KB[i])):

			KB_enc[i][j] = model.encode(KB[i][j])

	user_arg_enc=model.encode(user_argument)

	return KB_enc, user_arg_enc


def similarity_extraction(KB_enc, user_arg_enc, similarity_function):

	KB_sim=np.array[len(KB_enc),len(KB_enc[i])]

	for i in range(len(KB_enc):
		for j in range(len(KB_enc[i]):

			KB_sim[i][j] = np.dot(KB_enc[i][j],user_arg_enc)


	return KB_sim





def argument_classification_ms(model, user_argument, KB, thereshold=0.8, similarity_function):


	KB_enc, user_arg_enc = text_encoder(model, user_argument, KB)

	KB_sim = similarity_extraction(KB_enc, user_arg_enc, similarity_function)

	













