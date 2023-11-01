import json
import urllib
from typing import List
from SPARQLWrapper import SPARQLWrapper, JSON

sparql = SPARQLWrapper("http://127.0.0.1:3001/sparql")
sparql.setReturnFormat(JSON)


with open('data/fb_roles', 'r') as f:
    contents = f.readlines()

roles = set()
for line in contents:
    fields = line.split()
    roles.add(fields[1])

def process_file(filename):
    data = json.load(open(filename, 'r'), strict=False)
    selected_data = []
    for example in data:
        sele_dict = {}
        sele_dict["qid"] = example["qid"]
        sele_dict["question"] = example["question"]
        sele_dict["function"] = example["function"]
        sele_dict["domains"] = example["domains"]
        sele_dict["sparql_query"] = example["sparql_query"]
        sele_dict["s_expression"] = example["s_expression"]
        sele_dict["answer"] = example["answer"]
        selected_data.append(sele_dict)
    return selected_data

def find_data_by_question(selected_data, question):
    for data_item in selected_data:
        if data_item["question"] == question:
            return data_item
    return None

def get_out_relations(entity: str):
    out_relations = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations

    
def get_in_relations(entity: str):
    in_relations = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations    
  

def get_types(entity: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)
    # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def get_out_relation_by_type(class_list,out_relations_list):
    class_relations = {class_name: set() for class_name in class_list}
    relation_list = []
    
    for line in contents:
        parts = line.strip().split()
        if len(parts) == 3:
            source_class, relation, target_class = parts
            if source_class in class_list and relation in out_relations_list:
                class_relations[source_class].add(relation)
                relation_list.append(relation)
    return relation_list

    
def get_in_relation_by_type(class_list,in_relations_list):
    class_relations = {class_name: set() for class_name in class_list}
    relation_list = []
    
    for line in contents:
        parts = line.strip().split()
        if len(parts) == 3:
            target_class, relation, source_class = parts
            if source_class in class_list and relation in in_relations_list:
                class_relations[source_class].add(relation)
                relation_list.append(relation)
    return relation_list


def get_in_type_by_relation(in_relation):
    for line in contents:
        parts = line.strip().split()
        if len(parts) == 3:
            target_class, relation, source_class = parts
            if relation == in_relation:
                return target_class
    return ''

def get_out_type_by_relation(in_relation):
    for line in contents:
        parts = line.strip().split()
        if len(parts) == 3:
            source_class, relation, target_class = parts
            if relation == in_relation:
                return target_class
    return ''

def get_out_relation_by_num(class_list):
    type_num_date_list = ['type.int','type.float','type.datetime']
    class_relations = {class_name: set() for class_name in class_list}
    relation_list = []

    for line in contents:
        parts = line.strip().split()
        if len(parts) == 3:
            source_class, relation, target_class = parts
            if source_class in class_list and target_class in type_num_date_list:
                class_relations[source_class].add(relation)
                relation_list.append(relation)
    return relation_list
