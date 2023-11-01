import json
import re
from rank_bm25 import BM25Okapi
import openai
import utils as util

# Set OpenAI API configuration
openai.api_key = "YOUR_OPENAI_KEY"

# Load KB schema
with open('data/fb_roles', 'r') as role_file:
    role_contents = role_file.readlines()
roles = set(field.split()[1] for field in role_contents)

# Load LSP result
lsp_result_file_path = 'result/lsp_result.json'
with open(lsp_result_file_path, 'r') as lsp_file:
    lsp_result = json.load(lsp_file)

# Load entity linking result
el_results_data = {}
el_results_file_path = 'data/grail_combined_tiara.json'
with open(el_results_file_path, 'r', encoding='utf-8') as el_file:
    el_results_data = json.load(el_file)

# Load question IDs
dev_qids_file_path = "data/qid2ques_dev.txt"
with open(dev_qids_file_path, "r", encoding='utf-8') as dev_file:
    dev_qids = [line.strip().split("\t")[0] for line in dev_file]

# Load answer types
answer_types_file_path = "data/answer_types_grail_combined.txt"
with open(answer_types_file_path, "r") as answer_file:
    answer_lines = answer_file.readlines()

# Create a dictionary mapping qid to class_type_list for non-test qids from answer_lines
qid2cls_dict = {parts[0]: parts[1:] for parts in (line.strip().split("\t") for line in answer_lines if not line.startswith("test"))}

# Process the sample data
sample_data = util.process_file("data/grailqa_v1.0_train.json")

# Preprocess data
qid2ques_sample_file_path = "data/qid2ques_train.txt"
with open(qid2ques_sample_file_path, "r", encoding='utf-8') as qid_file:
    sample_questions = [line.strip().split("\t")[-1] for line in qid_file]

# Build corpus and calculate BM25 model
corpus = [question.split() for question in sample_questions]
bm25 = BM25Okapi(corpus)

# Function to get entity link
def get_entity_link(qid):
    el_results_item = el_results_data.get(qid, {})
    for m in el_results_item:
        for mid in el_results_item[m]:
            return mid
    return None

# Function to get samples
def get_samples(query, skeleton, candidates, top_k):
    # Split the input query into tokens
    query_tokens = query.split()

    # Calculate BM25 scores for the query tokens
    bm25_scores = bm25.get_scores(query_tokens)

    # Determine the number of similar questions to retrieve
    num_similar_questions = top_k

    # Sort question indices based on BM25 scores and select top-k similar questions
    similar_question_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:num_similar_questions])

    # Extract similar questions based on their indices
    similar_questions = [sample_questions[i] for i in similar_question_indices]

    # Initialize a variable to store the generated ICL samples
    icl_samples = ""

    # Iterate through the similar questions
    for i, question in enumerate(similar_questions, start=1):
        # Find sample data based on the question
        found_data = util.find_data_by_question(sample_data, question)

        # Get the S-expression from the found data
        s_expression = found_data.get("s_expression", "")

        # Define a regular expression pattern to extract the skeleton
        pattern = r'(\b(?:AND|JOIN|COUNT|ARGMAX|ARGMIN|le|lt|ge|gt|R)\b|[()])'

        # Extract the skeleton from the S-expression
        extracted_skeleton = ''.join(re.findall(pattern, s_expression)).replace("(R)", "")

        if extracted_skeleton == skeleton:
            # print(question, s_expression)
            sel_result = ""
            if "ARG" in skeleton:
                sel_result = s_expression.split(" ")[-1].strip(")")
            if "(l" in skeleton or "(g" in skeleton:
                sel_result = s_expression.split(" ")[-2]
            if "COUNT" in skeleton:
                sel_result = s_expression.split(" ")[-2].strip(")")
            if "(AND(JOIN))" == skeleton:
                sel_result = s_expression.split(" ")[-2].strip(")")
            if "(AND(JOIN(JOIN)))" == skeleton:
                sel_result_1 = s_expression.split(" ")[-2].strip(")")
                sel_result_2 = s_expression.split("(JOIN")[1].strip("(R ").strip(") ")
                sel_result = sel_result_1 + "," + sel_result_2
            if sel_result in candidates:
                icl_samples = icl_samples + f"""
                Question: {question}
                Relation: {sel_result}"""


    # Return the generated ICL samples
    return icl_samples

# Function to get LLM relation
def get_llm_rel(query, relation_list, icl_samples):
    # Define the system prompt that instructs the model to select the most relevant relationship
    system_prompt = """
    Please select the most relevant relationship from the relationship list for the following question, and only return the relationship name without any extra content.
    Note: The returned relationship name must be a real relationship that can be found in the relationship list.
    """

    # Create a user prompt that presents the question and the available relationship list
    user_prompt = f"""
    Question: {query}
    Type: {relation_list}
    """

    # If there are ICL samples available, append them to the system prompt
    if icl_samples:
        system_prompt += icl_samples

    # Define a list of messages for the model, including both system and user messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Generate completions from the model based on the provided prompts
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=2048,
        messages=messages,
    )

    # Extract the response from the model's completion
    response = completions.choices[0].message.content

    # Extract and clean the best matching relation from the response
    best_match_relation = response.replace("Relation:", "").replace("[", "").replace("]", "").replace("'", "")

    # Return the best matching relation
    return best_match_relation

result_list = []

for item in lsp_result:
    # Extract relevant information from the current item
    index = item["index"]   # Index of the current item
    qid = dev_qids[index]   # Query ID associated with the index
    query = item["query"]   # The target question
    logical_type = item["logical_type"]  # The logical type associated with the query
    skeleton = item["skeleton"] # The skeleton associated with the query

    # Obtain entity information linked to the query
    entity_mid = get_entity_link(qid)

    # Set the value for 'top_k' of sample questions
    top_k = 40

    if logical_type == "Count":
        # Obtain a list of types associated with the entity and Filter and create a list of relevant classes
        types = util.get_types(entity_mid)
        class_list = [cur_type for cur_type in types if not cur_type.startswith("common.") and not cur_type.startswith("base.")]

        # Retrieve the incoming relations for the entity and Filter incoming relations based on class_list
        in_relations = util.get_in_relations(entity_mid)
        in_relation_list = util.get_in_relation_by_type(class_list, in_relations)

        # Retrieve the outgoing relations for the entity and Filter outgoing relations based on class_list
        out_relations = util.get_out_relations(entity_mid)
        out_relation_list = util.get_out_relation_by_type(class_list, out_relations)

        # Combine incoming and outgoing relations into a single list
        all_relation_list = in_relation_list + out_relation_list

        # Generate ICL samples based on the query, skeleton, and the combined relation list
        icl_samples = get_samples(query, skeleton[0], all_relation_list, top_k)

        # Use LLM to determine the best matching relation from the relation list
        best_match_relation = get_llm_rel(query, all_relation_list, icl_samples)

        # Initialize target_class and exp_Logic_Skeleton variables
        target_class = ""
        exp_Logic_Skeleton = ""

        # Check if the best matching relation is an incoming relation
        if best_match_relation in in_relations:
            # Obtain the target class associated with the relation and Define an expression for the logical skeleton
            target_class = util.get_in_type_by_relation(best_match_relation)
            exp_Logic_Skeleton = "(COUNT (AND <class> (JOIN <relation> <entity>)))"

        # Check if the best matching relation is an outgoing relation
        if best_match_relation in out_relations:
            # Obtain the target class associated with the relation and Define an expression for the logical skeleton
            target_class = util.get_out_type_by_relation(best_match_relation)
            exp_Logic_Skeleton = "(COUNT (AND <class> (JOIN (R <relation>) <entity>)))"

        # Substitute placeholders in the logical skeleton with actual values
        sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation).replace("<entity>", entity_mid)
        print(sel_Logic_Expression)

    elif logical_type == "Superlative":
        # Retrieve the class list associated with the query ID
        class_list = qid2cls_dict[qid]

        # Get a list of outgoing relations based on the numeric values associated with the class list
        all_relation_list = util.get_out_relation_by_num(class_list)

        # Generate ICL samples based on the query, skeleton, and the combined relation list
        icl_samples = get_samples(query, skeleton[0], all_relation_list, top_k)

        # Use LLM to determine the best matching relation from the relation list
        best_match_relation = get_llm_rel(query, all_relation_list, icl_samples)

        # Obtain the target class associated with the best matching relation
        target_class = util.get_in_type_by_relation(best_match_relation)

        # Check if the skeleton contains "MAX"
        if "MAX" in skeleton[0]:
            # Define an expression for the logical skeleton for "ARGMAX"
            exp_Logic_Skeleton = "(ARGMAX <class> <relation>)"
            # Substitute placeholders in the logical skeleton with actual values
            sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation)
        # Check if the skeleton contains "MIN"
        if "MIN" in skeleton[0]:
            # Define an expression for the logical skeleton for "ARGMIN"
            exp_Logic_Skeleton = "(ARGMIN <class> <relation>)"
            # Substitute placeholders in the logical skeleton with actual values
            sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation)
        print(sel_Logic_Expression)

    elif logical_type == "Comparison":
        # Retrieve the class list associated with the query ID
        class_list = qid2cls_dict[qid]

        # Get a list of outgoing relations based on the numeric values associated with the class list
        all_relation_list = util.get_out_relation_by_num(class_list)

        # Generate ICL samples based on the query, skeleton, and the combined relation list
        icl_samples = get_samples(query, skeleton[0], all_relation_list, top_k)

        # Use LLM to determine the best matching relation from the relation list
        best_match_relation = get_llm_rel(query, all_relation_list, icl_samples)

        # Obtain the target class associated with the best matching relation
        target_class = util.get_in_type_by_relation(best_match_relation)

        # Define a mapping from skeleton conditions to corresponding logical conditions
        condition_map = {'lt': 'lt', 'le': 'le', 'gt': 'gt', 'ge': 'ge'}

        # Use the 'get' method to extract a logical condition from the skeleton, if it exists
        gen_lget = condition_map.get(next((condition for condition in condition_map if condition in skeleton[0]), ''))

        # Define a logical skeleton for conditional expressions
        exp_Logic_Skeleton = "(AND <class> (<lget> <relation> <literal>)"

        # Substitute placeholders in the logical skeleton with actual values
        sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation).replace("<literal>", entity_mid).replace("<lget>", gen_lget)
        print(sel_Logic_Expression)

    elif logical_type == "General":
        # Check if the skeleton contains "(AND(JOIN))" and the entity_mid doesn't contain liretal
        if "(AND(JOIN))" in skeleton and "^^" not in entity_mid:
            # Obtain a list of types associated with the entity and Filter and create a list of relevant classes
            types = util.get_types(entity_mid)
            class_list = [cur_type for cur_type in types if not cur_type.startswith("common.") and not cur_type.startswith("base.")]

            # Retrieve the incoming relations for the entity and Filter incoming relations based on class_list
            in_relations = util.get_in_relations(entity_mid)
            all_relation_list = util.get_in_relation_by_type(class_list, in_relations)

            # Generate ICL samples based on the query, skeleton, and the combined relation list
            icl_samples = get_samples(query, skeleton[0], all_relation_list, top_k)
            # Use LLM to determine the best matching relation from the relation list
            best_match_relation = get_llm_rel(query, all_relation_list, icl_samples)

            # Obtain the target class associated with the best matching relation
            target_class = util.get_in_type_by_relation(best_match_relation)

            # Define a logical skeleton for the "General" case
            exp_Logic_Skeleton = "(AND <class> (JOIN <relation> <entity>))"
            # Substitute placeholders in the logical skeleton with actual values
            sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation).replace("<entity>", entity_mid)
            print(sel_Logic_Expression)

        elif "(AND(JOIN))" in skeleton and "^^" in entity_mid:
            # Retrieve the class list associated with the query ID
            class_list = qid2cls_dict[qid]
            # Get a list of outgoing relations based on the numeric values associated with the class list
            all_relation_list = util.get_out_relation_by_num(class_list)

            # Generate ICL samples based on the query, skeleton, and the combined relation list
            icl_samples = get_samples(query, skeleton[0], all_relation_list, top_k)
            # Use LLM to determine the best matching relation from the relation list
            best_match_relation = get_llm_rel(query, all_relation_list, icl_samples)

            # Obtain the target class associated with the best matching relation
            target_class = util.get_in_type_by_relation(best_match_relation)

            # Define a logical skeleton for the "General" case
            exp_Logic_Skeleton = "(AND <class> (JOIN <relation> <entity>))"
            # Substitute placeholders in the logical skeleton with actual values
            sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<relation>", best_match_relation).replace("<entity>", entity_mid)
            print(sel_Logic_Expression)

        # Check if the skeleton contains "(AND(JOIN(JOIN)))"
        if "(AND(JOIN(JOIN)))" in skeleton:
            # Obtain a list of types associated with the entity and Filter and create a list of relevant classes
            types = util.get_types(entity_mid)
            class_list = [cur_type for cur_type in types if not cur_type.startswith("common.") and not cur_type.startswith("base.")]

            # Retrieve the incoming relations for the entity and Filter incoming relations based on the class list
            one_hop_in_relations = util.get_in_relations(entity_mid)
            one_hop_in_relation_list = util.get_in_relation_by_type(class_list, one_hop_in_relations)
            # Retrieve the outgoing relations for the entity and Filter outgoing relations based on the class list
            one_hop_out_relations = util.get_out_relations(entity_mid)
            one_hop_out_relation_list = util.get_out_relation_by_type(class_list, one_hop_out_relations)
            # Combine one-hop incoming and outgoing relations into a single list
            one_hop_all_relation_list = one_hop_in_relation_list + one_hop_out_relation_list

            # Generate ICL samples based on the query, skeleton, and the combined one-hop relation list
            icl_samples = get_samples(query, skeleton[0], one_hop_all_relation_list, top_k)
            # Use LLM to determine the best matching relation from the one-hop relation list
            one_hop_best_match_relation = get_llm_rel(query, one_hop_all_relation_list, icl_samples)

            # Check if the best matching relation is in the one-hop outgoing relations
            if one_hop_best_match_relation in one_hop_out_relations:
                # Obtain the one-hop class based on the best matching relation
                one_hop_class = util.get_out_type_by_relation(one_hop_best_match_relation)
            # Check if the best matching relation is in the one-hop incoming relations
            elif one_hop_best_match_relation in one_hop_in_relations:
                # Obtain the one-hop class based on the best matching relation
                one_hop_class = util.get_in_type_by_relation(one_hop_best_match_relation)

            # Retrieve two-hop incoming relations based on the one-hop class
            two_hop_in_relations = util.get_in_relation_by_class(one_hop_class)
            # Retrieve two-hop outgoing relations based on the one-hop class
            two_hop_out_relations = util.get_out_relation_by_class(one_hop_class)
            # Combine two-hop incoming and outgoing relations into a single list
            two_hop_all_relations = two_hop_in_relations + two_hop_out_relations

            # Generate ICL samples based on the query, skeleton, and the combined two-hop relation list
            icl_samples = get_samples(query, skeleton[0], two_hop_all_relations, top_k)
            # Use LLM to determine the best matching relation from the two-hop relation list
            two_hop_best_match_relation = get_llm_rel(query, two_hop_all_relations, icl_samples)

            # Check if the best matching relation is in the two-hop incoming relations
            if two_hop_best_match_relation in two_hop_in_relations:
                # Obtain the target class based on the best matching relation
                target_class = util.get_in_type_by_relation(two_hop_best_match_relation)
            # Check if the best matching relation is in the two-hop outgoing relations
            if two_hop_best_match_relation in two_hop_out_relations:
                # Obtain the target class based on the best matching relation
                target_class = util.get_out_type_by_relation(two_hop_best_match_relation)

            # Define various logical skeletons for different combinations of one-hop and two-hop relations
            if one_hop_best_match_relation in one_hop_in_relations and two_hop_best_match_relation in two_hop_in_relations:
                exp_Logic_Skeleton = "(AND <class> (JOIN <two_hop_relation>) (JOIN <one_hop_relation> <entity>)))"
            if one_hop_best_match_relation in one_hop_in_relations and two_hop_best_match_relation in two_hop_out_relations:
                exp_Logic_Skeleton = "(AND <class> (JOIN <two_hop_relation>) (JOIN (R <one_hop_relation>) <entity>)))"
            if one_hop_best_match_relation in one_hop_out_relations and two_hop_best_match_relation in two_hop_in_relations:
                exp_Logic_Skeleton = "(AND <class> (JOIN (R <two_hop_relation>) (JOIN <one_hop_relation> <entity>)))"
            if one_hop_best_match_relation in one_hop_out_relations and two_hop_best_match_relation in two_hop_out_relations:
                exp_Logic_Skeleton = "(AND <class> (JOIN (R <two_hop_relation>) (JOIN (R <one_hop_relation>) <entity>)))"

            # Substitute placeholders in the logical skeleton with actual values
            sel_Logic_Expression = exp_Logic_Skeleton.replace("<class>", target_class).replace("<one_hop_relation>", one_hop_best_match_relation).replace("<two_hop_relation>", two_hop_best_match_relation).replace("<entity>", entity_mid)
            print(sel_Logic_Expression)

    # Save the result to the result list
    result_dict = {"index": index, "qid": qid, "query": query, "type": type, "skeleton": skeleton, "logic_form": sel_Logic_Expression}
    result_list.append(result_dict)
    print(result_dict)

# Save results to a JSON file
with open("result/kcr_result.json", "w") as json_file:
    json.dump(result_list, json_file, indent=4)
