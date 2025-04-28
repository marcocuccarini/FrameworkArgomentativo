import json

def load_directed_graph(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    graph = {}
    indice=0   ########visto che sono tanti grafi per controllare ogni albero si deve definire il rispettivo indice. 
    for attack in data[indice].get('attacks', []):
        source = attack['source']
        target = attack['target']
        graph.setdefault(source, []).append(target)
        graph.setdefault(target, [])  # assicura che anche i nodi solo target siano presenti
    return graph

def has_cycle_directed(graph):
    visited = set()
    stack = set()

    def visit(node):
        if node in stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        stack.add(node)
        for neighbor in graph.get(node, []):
            if visit(neighbor):
                return True
        stack.remove(node)
        return False

    for node in graph:
        if visit(node):
            return True
    return False

# Uso
graph = load_directed_graph('arg_dic.json')
if has_cycle_directed(graph):
    print("Il grafo ORIENTATO contiene almeno un ciclo.")
else:
    print("Il grafo ORIENTATO NON contiene cicli.")