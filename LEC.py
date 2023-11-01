import json
import openai

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_KEY"

def get_llm_answer_type(query, answer_type_list):
    # Define the system prompt
    system_prompt = f"""
    Please select an answer type from the list that is most suitable as the answer to the question.
    Note: Please directly provide the answer type and do not return any extra content.
    Note: The answer type returned must be an element that exists in the list.
    """

    # Create a user prompt with the provided question and answer type list
    user_prompt = f"""
    Question：{query}
    AnswerType：{answer_type_list}
    """

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
        messages=messages,
    )
    response = (completions).choices[0].message.content

    # Extract the best answer type from the response
    best_answer_type = response.replace("AnswerType：", "").replace("[", "").replace("]", "").replace("'", "")

    return best_answer_type



def main():

    # Read the KCR result
    with open('result/kcr_result.json', 'r') as file:
        kcr_result = json.load(file)

    results_list = []
    for item in kcr_result:
        qid = item["qid"]
        query = item["query"]
        logical_type = item["logical_type"]
        logical_expressions = item["logic_form"]

        if logical_type == "General":
            # Split each logical expression to extract the answer type (second word)
            answer_type_list = [le.split(" ")[1] for le in logical_expressions]

            # Create a dictionary (mapping) that associates each answer type with its corresponding logical expression
            at2le = {answer_type: le for le, answer_type in zip(logical_expressions, answer_type_list)}

            # Get the best answer type using the LLM model
            best_answer_type = get_llm_answer_type(query, answer_type_list)

            # Retrieve the logical expression corresponding to the best answer type from the dictionary
            logical_expression = at2le[best_answer_type]
        else:
            logical_expression = logical_expressions

        # Save the result to the result list
        result_dict = {"qid": qid, "query": query, "logic_form": logical_expression}
        results_list.append(result_dict)
        print(result_dict)

    # Save results to a JSON file
    with open("result/lsc_result.json", "w") as json_file:
        json.dump(results_list, json_file, indent=4)

if __name__ == "__main__":
    main()
