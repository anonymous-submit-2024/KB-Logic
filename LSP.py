import json
import re
from collections import Counter
from rank_bm25 import BM25Okapi
import openai
import utils as util

# Set OpenAI API configuration
openai.api_key = "YOUR_OPENAI_KEY"

# Set top-n of questions
top_n = 100

# Set top-k of skeletons
top_k = 5


# Function to get logical type using OpenAI
def get_logical_type(query):
    # Define the system prompt
    system_prompt = """
        Please classify the following question and return only the classification and do not return any other content.
        Question: the biggest catchment area can be found at what lake? 
        Type:{General,Count,Superlative,Comparison}
        """

    # Create a user prompt with the target question
    user_prompt = f"Question:{query}"

    # Construct a list of messages for the OpenAI API input
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Generate completions using the OpenAI model
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=2048,
        messages=messages
    )
    response = (completions).choices[0].message.content
    logical_type = response.replace("Type", "").replace(":", "").strip()
    return logical_type

# Function to filter skeletons based on logical type
def filter_skeleton(logical_type, expressions):
    skeletons = []

    count_type = ['COUNT']
    sup_type = ['ARGMAX', 'ARGMIN']
    comp_type = ['le', 'lt', 'ge', 'gt']

    for ske in expressions:
        if (logical_type == "General" and not any(item in ske for item in count_type + sup_type + comp_type)) or \
           (logical_type == "Count" and any(item in ske for item in count_type)) or \
           (logical_type == "Superlative" and any(item in ske for item in sup_type)) or \
           (logical_type == "Comparison" and any(item in ske for item in comp_type)):
            skeletons.append(ske)

    return skeletons

def main():
    # Process the sample data
    sample_data = util.process_file("data/grailqa_v1.0_train.json")

    # Load the sample questions
    with open("data/qid2ques_train.txt", "r", encoding='utf-8') as file:
        sample_questions = [line.strip().split("\t")[-1] for line in file]

    # Load the target questions
    with open("data/qid2ques_dev.txt", "r", encoding='utf-8') as file:
        target_questions = [line.strip().split("\t")[-1] for line in file]

    # Build corpus and the BM25 model
    corpus = [question.split() for question in sample_questions]
    bm25 = BM25Okapi(corpus)

    results_list = []

    for index, query in enumerate(target_questions):
        # Split the input query into tokens
        query_tokens = query.split()

        # Calculate BM25 scores for the query tokens
        bm25_scores = bm25.get_scores(query_tokens)

        # Sort question indices based on BM25 scores and select top-k similar questions
        similar_question_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_n]

        # Extract similar questions based on their indices
        similar_questions = [sample_questions[i] for i in similar_question_indices]

        # Extract Logical Skeletons
        logical_skeletons = []
        for i, question in enumerate(similar_questions, start=1):
            # Find sample data based on the question
            found_data = util.find_data_by_question(sample_data, question)

            # Get the S-expression from the found data
            s_expression = found_data["s_expression"]

            # Define a regular expression pattern to extract the skeleton
            pattern = r'(\b(?:AND|JOIN|COUNT|ARGMAX|ARGMIN|le|lt|ge|gt|R)\b|[()])'

            # Extract the skeleton from the S-expression
            extracted_skeletons = ''.join(re.findall(pattern, s_expression)).replace("(R)", "")
            logical_skeletons.append(extracted_skeletons)

        # Count their occurrences
        skeleton_counts = Counter(logical_skeletons)

        # Sort expression types by count
        sorted_expression_types = sorted(skeleton_counts, key=lambda x: skeleton_counts[x], reverse=True)

        # Extract top-k logical skeletons
        top_k = 5
        top_k_skeletons = sorted_expression_types[:top_k]

        # Filter skeletons based on logical type
        logical_type = get_logical_type(query)
        skeletons = filter_skeleton(logical_type, top_k_skeletons)

        # Save the result to the result list
        result_dict = {"index": index, "query": query, "logical_type": logical_type, "skeleton": skeletons}
        results_list.append(result_dict)
        print(result_dict)

    # Save results to a JSON file
    with open("result/lsp_result.json", "w") as json_file:
        json.dump(results_list, json_file, indent=4)

if __name__ == "__main__":
    main()
