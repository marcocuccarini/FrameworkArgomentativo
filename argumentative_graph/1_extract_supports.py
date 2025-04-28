def check_result(matrix_support):
    result = set()
    checked = set()
    
    for key, value in matrix_support.items():
        i, j = key
        if (j, i) in matrix_support and (j, i) not in checked:
            try:
                val1 = float(matrix_support[key])
            except ValueError:
                val1 = None
            try:
                val2 = float(matrix_support[(j, i)])
            except ValueError:
                val2 = None
            
            # Entrambi numeri
            if val1 is not None and val2 is not None:
                if val1 > 0.8 and val2 > 0.8:
                    result.add(key if val1 >= val2 else (j, i))
                elif val1 > 0.8:
                    result.add(key)
                elif val2 > 0.8:
                    result.add((j, i))
            # Solo uno è numero
            elif val1 is not None and val1 > 0.8:
                result.add(key)
            elif val2 is not None and val2 > 0.8:
                result.add((j, i))
    
            checked.add(key)
            checked.add((j, i))
        
        elif (j, i) not in matrix_support:
            try:
                val = float(value)
                if val > 0.8:
                    result.add(key)
            except ValueError:
                pass
    
    return result

def get_supports(tupla, dict_tree):

    for key in tupla:
        
        triple = {}
        
        if len(tupla[key]) > 1 and len(tupla[key]) < 4:
            
            for item in tupla[key]:
    
                triple[item['id']] = item['sentences'][0]
                
            matrix_support=matrix_creation.calculate_matrix(triple)
            
            result=check_result(matrix_support)


            print(result)
            
            dictionary_support = {key: [] for key in dict_tree}

            for key in dict_tree.keys():

                for i in result:
                    
                    if list(i)[0] in dict_tree[key] and list(i)[1] in dict_tree[key]:



        
                        dictionary_support[key].append({"source": list(result)[0][0], "target": list(result)[0][1], "weight": 1})

    
    return dictionary_support

import sys
print(sys.version)
import ollama
import pandas as pd
from pathlib import Path
from llms_classes import *
from prompt_classes import *
from dataset_classes import TextDataset
from matrix_creation_classes import *


text_dataset=TextDataset("/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/test_text_class.csv")

text_dataset.filter_dataset_notclass()

prompt=PromptCreation(False, "/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/test_prompts.csv","/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/Argument_Mining/dataset_input/samples_fewshot.csv")

resulting_prompt=prompt.prompt_creation()


MODEL = "llama3.1:latest"
#MODEL= "mistral"
#MODEL = "qwen2.5"

# Init the ollama server
ollama_server = OllamaServer(ollama)

# Check models that have already been downloaded
models = ollama_server.get_models_list()
print("Available Models:", models)

# Download the model to use for experiences
models = ollama_server.download_model_if_not_exists(MODEL)

# Initialize the client and server
ollama_server = OllamaServer(ollama)

# Initialize chat with a specific model
chat = OllamaChat(server=ollama_server, model=MODEL)

#create the object for the creation of the support matrix
matrix_creation=matrix_creation(chat, resulting_prompt)

#this part of the code get all the possible situation to apply the propose approche:
'''
B -> A
C -> A
... -> A

For this reason we get the dictionary arg_dic, and we goup by by previus id. That mean that when two argument share 
the same "previus_id" it means that attacks the same argument. 
That are group by into according prev_id, this tuple have varius size, (from 2 to n sentences).

Dict["prev_id"] -> [{argument that attack the same id}]


'''

obj=triple_extraction("/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/arg_dic.json")

tupla=obj.extract_list_tuple()

#dict tree keeps trac

#tutte le tuple, organizzate per id comun eattaccato, selezione anche il dizionario che mi tiene traccia dell'appartenznza
# di tutti gli argument al relativo albero [indicato dall'indice da 0 a numero di alberi totale. 

with open('/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/dictionary_tree.json', 'r', encoding='utf-8') as file:
    dict_tree = json.load(file)



'''
#la tupla e il dict_tree vengono passati la tupla, esempio di una tupla: { sentenceA, sentenceB, sentenceC } 
# la funzione get_supports va applicare a tutte le tuple estratte:

#dato in in put a list of tuple
[{ sentenceA, sentenceB, sentenceC }, ...] -> [[       senA,  senB,   sencC   
                                               senA,   X      0.5    0.9
                                               senB,   0.8     X     0.7
                                               senC     0.9    0.1    X
                                               ],
                                               ....]



[[           senA,  senB,   sencC   
       senA,   X      0.5    0.9
       senB,   0.8     X     0.7    --->      [[[SenA support SenC],[SenB support sen A]],....]
       senC     0.1    0.1    X
       ],
       ....]
'''

#This function extract 
import json
with open('/Users/marco/Documents/GitHub/FrameworkArgomentativo/argumentative_graph/arg_dic.json', 'r', encoding='utf-8') as file:
    dict_arg = json.load(file)


#this part of the code add to the arg dic all the supports extaracted
supports=get_supports(tupla, dict_tree)

for i in supports:

    for j in dict_arg:

        if dict_arg['_id'] == i:

            dict_arg['supports'] = supports[i]

            
with open('arg_dic_supp.json', 'w') as f:
    json.dump(dict_arg, f)