import sys
print(sys.version)
import ollama
import pandas as pd
from pathlib import Path
from llms_classes import *
from prompt_classes import *
from dataset_classes import TextDataset


text_dataset=TextDataset("/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/test_text_class.csv")


text_dataset.filter_dataset_notclass()


prompt=PromptCreation(False, "/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/test_prompts.csv","/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/samples_fewshot.csv")


resulting_prompt=prompt.prompt_creation()

#create the prompt, the False provide no few shot learning


# Init the ollama server

MODEL = "llama3.1:latest"
ollama_server = OllamaServer(ollama)

# Check models that have already been downloaded
models = ollama_server.get_models_list()

#Print all the avaible Ollama model dowload
#print("Available Models:", models)

# Download the model to use for experiences
models = ollama_server.download_model_if_not_exists(MODEL)

# Initialize the client and server
ollama_server = OllamaServer(ollama)

# Initialize chat with a specific model
chat = OllamaChat(server=ollama_server, model=MODEL)


matrix_creation=matrix_creation(chat, prompt)


triple = {
   "a": "A degree is important but not necessary, often companies also want people that have better work ethics and more work experience instead of academic knowledge.",
   "b": "I don't think that the number of students will be significantly reduced by paid studies. If someone is determined to study then he will find financial resources for the payment even if he has to find a float or take a loan",
   "c": "Education is important not only from an individual point of view, but also for society. Everyone should be encouraged to go to university. Maybe student loans need to be reframed to the student saying thank you when they can afford to do so."
}

matrix_support=matrix_creation.calculate_matrix(triple)
obj=triple_extraction("/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph(arg_dic.json")

obj.extract_list_tuple()

