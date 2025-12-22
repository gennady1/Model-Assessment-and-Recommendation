'''This class will use an LLM to create n interpretations of statistical results. then will use ROUGE F1 score to create a confusion metrics, and measure information_retention.

Updates:
 2025-10-25 - Added deepseek model to the list.
 2025-12-05 - Renamed from 'metrics_llm_interpretation_nlp_confusion_matrix.py' to 'metrics_llm_information_retention.py'
            - Changed output folder name from 'interpretation_results' to 'information_retention'
''' 
import os
import json
import ollama
import evaluate
import itertools

# visualization
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


#--------Default parameters-------

DEBUG_LOG = True
# DEBUG_LOG = False

SAMPLE_COUNT = 10                        # number of interpretation samples for each LLM
MAX_OUTPUT_STRING_LENTH = 50             # used to limit the text length to display

# LLM Parameters
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.05

# for debug purposes
# LLM_LIST = ['phi3:3.8b']
# LLM_LIST = ['phi3:3.8b', 'gemma3:4b']

LLM_LIST = [
'phi3:3.8b', 'gemma3:4b', 'codellama:7b', 'mistral:7b','qwen3:8b', 'deepseek-r1:8b',         # Thinking models: 'qwen3:8b', 'deepseek-r1:8b',      
'mistral-nemo:12b', 'gemma3n:e4b', 'gemma3:12b-it-qat', 'phi4:14b', 'codestral:22b',
'zephyr:7b-beta-fp16', 'gemma3:27b', 'gemma3:27b-it-qat', 'llama3.3:70b']

# STATS_LIST = ['dataframe']
STATS_LIST = ['dataframe', 'correlation', 'linear-regression', 'multicollinearity', 'pca', 'combined']

#create output directory if needed
cm_folder = "confusion_matrix"
cm_path = os.path.join(os.getcwd(), cm_folder)
if not os.path.exists(cm_path):
    os.makedirs(cm_path)
    print(f"\n Output directory for confusion_matrix: {cm_path}")

#create directory for storing interpretation results; need to change this 
interpretation_folder = "interpretation_results"
intrp_path = os.path.join(os.getcwd(), interpretation_folder)
if not os.path.exists(intrp_path):
    os.makedirs(intrp_path)
    print(f"\n Output directory for confusion_matrix: {interpretation_folder}")

def pretty_print_json(json_obj):
    print(json.dumps(json_obj, indent=4))

def _compute_bertscore(interpretation, reference):
    try:
        r = evaluate.load('bertscore').compute(lang="en", predictions=[interpretation], references=[reference])
        return r
    except Exception as e:
        print(f"\n\tException while computing BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def plot_confusion_matrix(conf_matrix, labels, title, output_file):
    # Convert the confusion matrix into a 2D list of floats
    conf_matrix = [[float(x[0]) if isinstance(x, list) else float(x) for x in row] for row in conf_matrix]

    sns.set(rc={'figure.figsize':(11.7, 8.27)})
    ax = sns.heatmap(conf_matrix, annot=True, fmt=".3f", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)

    plt.title(title)
    # plt.xlabel('Predicted')
    # plt.ylabel('True')

    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")
    plt.close('all')
    # plt.clf()
    # plt.show()

# def plot_confusion_matrix(conf_matrix, labels, title, output_file):
#     # Convert the confusion matrix into a 2D list of floats
#     conf_matrix = [[float(x[0]) if isinstance(x, list) else float(x) for x in row] for row in conf_matrix]

#     plt.imshow(conf_matrix, interpolation='nearest', cmap="YlGnBu")
#     plt.title(title)
#     plt.colorbar()  # This will create a single color bar
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)

#     # Convert the confusion matrix into a NumPy array
#     conf_matrix = np.array(conf_matrix)
#     thresh = conf_matrix.max() / 2.
#     for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
#         plt.text(j, i, "{:.2f}".format(conf_matrix[i, j]),
#                  horizontalalignment="center",
#                  color="white" if conf_matrix[i, j] > thresh else "black")

#     plt.tight_layout()
#     # plt.ylabel('True label')
#     # plt.xlabel('Predicted label')
#     plt.savefig(output_file)


# Call LLM
def _llm_call(system_prompt, input_text, model):
    response = ollama.chat(model=model, messages=[ {
                'role': 'system',
                'content': system_prompt,     #1. set model context 
            }, {
                'role': 'user',
                'content': input_text
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )

    #metric call here
    return response['message']['content']

def pairwise_bert_similarity(word_list):
    """
    Compute similarity for each unique pair of words.

    Parameters
    ----------
    word_list : list[str]

    Returns
    -------
    list[tuple[int, int, float]]
        Each tuple is (word_index_i, word_index_j, similarity) with i < j.
    """
    similarities = []
    n = len(word_list)
    for i in range(n - 1):
        for j in range(i + 1, n):
            score = _compute_bertscore(word_list[i], word_list[j])['f1'][0]
            similarities.append((i, j, score))

    return similarities

def calculate_bert_similarities_with_input(llm_input_prompt, interpretion_list):
    """ Computes similaries between two strings: ll_input_prompt and each interpretation in the interpretion_list """
    similarities = []
    i=0
    for intpr in interpretion_list:
        score = _compute_bertscore(intpr, llm_input_prompt)['f1'][0]
        similarities.append(('i', i, score))
        i+=1

    return similarities

def plot_pairwise_scores(score_lists, labels=None, title='Boxplot', file_plot='boxplot.png'):
    """
    Create a box‑plot for any number of pairwise‑similarity result sets.

    Parameters
    ----------
    score_lists : list[list[tuple[int, int, float]]]
        Each element is the output of `pairwise_similarity()` – a list of
        (i, j, similarity) tuples.
    labels : list[str] | None
        Optional list of labels to use on the x‑axis.  If omitted, the
        indices of the score sets are used.

    Example
    -------
    >>> word_list = random_text(5)
    >>> s1 = pairwise_similarity(word_list)
    >>> s2 = pairwise_similarity(word_list)
    >>> plot_pairwise_scores([s1, s2], labels=['run1', 'run2'])
    """
    # 1️  Extract only the similarity values
    data = [[score for _, _, score in lst] for lst in score_lists]

    # 2️  Compute stats for each list
    def _stats(arr):
        arr = np.array(arr)
        return {
            "min":  np.min(arr),
            "max":  np.max(arr),
            "mean": np.mean(arr),
            "std":  np.std(arr),
        }

    stats_list = [_stats(d) for d in data]

    # 3️  Plot
    fig = plt.figure(figsize=(8, 5))
    plt.boxplot(
        data,
        tick_labels=labels if labels is not None else [f"set {i}" for i in range(len(data))],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="darkblue"),
        medianprops=dict(color="red"),
    )

    # Add mean ± 1 σ annotation
    for i, st in enumerate(stats_list, start=1):
        plt.text(
            i,
            st["mean"] + st["std"] * 0.2,
            f"μ={st['mean']:.3f}",  #\nσ={st['std']:.2f}
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.title(title)
    plt.ylabel("Similarity")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # save to file
    plt.savefig(file_plot)

    # plt.show()
    # fig.close()

def calculate_confusion_matrix_for_each_model(llm_responses):
    for model in llm_responses.keys():
        interpretations = llm_responses[model]
        n = len(interpretations)

        # Initialize the confusion matrix with zeros
        conf_matrix = [[0.0] * n for _ in range(n)]

        # Calculate pairwise BERTScore and update the confusion matrix
        for i in range(n):
            ref = interpretations[i]
            for j in range(n):
                if i != j:
                    comparison_result = _compute_bertscore(interpretation=ref, reference=interpretations[j])
                    conf_matrix[i][j] = comparison_result['f1']
                else:
                    conf_matrix[i][j] = 1.0  # Set diagonal elements to 1.0

    return conf_matrix


def _create_confusion_matrix_from_statistical_interpretation_results(sim_data_llm_results, sim_variable):
    '''
    '''

    sys_prompt = {
        'dataframe': 'You are an expert data statistician. You are analyzing a dataset’s summary statistics. Interpret the mean, standard deviation, min, 25%, 50%, 75%, and max values to understand the central tendency, dispersion, and distribution shape of the data.',
        'correlation': 'You are an expert data statistician. Given a correlation matrix, analyze the strength and direction of linear relationships between variables. Identify strongly correlated pairs (both positive and negative) and discuss implications for further analysis or modeling.',
        'linear-regression': 'You are an expert data statistician. Interpret the results of a simple linear regression, including the slope, intercept, and p-value. Discuss the significance of the relationship, the predictive power of the model, and any limitations or potential next steps in analysis.',
        'multicollinearity': 'You are an expert data statistician. Analyze the variance inflation factors (VIFs) to assess multicollinearity among variables. Identify variables with high VIF values, discuss the implications of multicollinearity on regression models, and suggest strategies for mitigating its effects.',
        'pca':'You are an expert data statistician. Given the results of a PCA, interpret the explained variance ratio of each principal component. Discuss how the data can be represented in a lower-dimensional space, identify the most contributing original variables to each component, and suggest potential applications or next steps.',
        'combined': 'You are an expert data statistician. Integrate the insights from summary statistics, correlation analysis, linear regression, multicollinearity assessment, and PCA. Synthesize these results to provide a comprehensive understanding of the dataset, including relationships between variables, potential issues with multicollinearity, and recommendations for data reduction or predictive modeling.'
    }

    # create a dictionary to save results
    llm_responces = {}
    for model in LLM_LIST:

        # #this would be place if you want to add llm input prompt to the confusion matrix prompt
        # llm_responces[model].append(sim_data_llm_results[sim_variable][stat]['phi3:3.8b']['llm-input'])

        for stat in STATS_LIST:

            #define file names
            output_file = cm_path+f"/confusion_matrix_{sim_variable}_{model.replace(':','-')}_{stat}.png"
            cf_file_name = cm_path+f"/confusion_matrix_{sim_variable}_{model.replace(':','-')}_{stat}.json"
            llm_input_prompt_file = cm_path+f"/input_prompt_interpt_{sim_variable}_{model.replace(':','-')}_{stat}.txt"

            # save statistical result input prompt to file
            llm_input_prompt = sim_data_llm_results[sim_variable][stat]['phi3:3.8b']['llm-input']
            with open(llm_input_prompt_file, "w", encoding="utf-8") as file:
                file.write(llm_input_prompt)

            #add model to dictionary
            llm_responces[model+stat] = []

            if os.path.isfile(cf_file_name):
                print(f"\n  File exists: {cf_file_name[:-3]} Skipping calculation.")

            else:
                print(f"\n  File does not exists: {cf_file_name[:-3]} Calculating results...")
                if DEBUG_LOG:
                    print(f"  Sim Variable: {sim_variable}, stat={stat}\n\tsys_prompt: {sys_prompt[stat].strip()[0:MAX_OUTPUT_STRING_LENTH]}...\n\tllm_input_prompt: {llm_input_prompt.strip()[0:MAX_OUTPUT_STRING_LENTH]}...")

                for i in range(SAMPLE_COUNT):
                    r = _llm_call(system_prompt=sys_prompt[stat], input_text=llm_input_prompt, model=model)
                    llm_responces[model+stat].append(r)

                if DEBUG_LOG:
                    print(f"Model: {model}; len={len(llm_responces[model+stat])}")
                    for i, r in enumerate(llm_responces[model+stat]):
                        print(f"\t{i}: {r.strip()[0:MAX_OUTPUT_STRING_LENTH]}...")
                #calculate confusion matrix
                confusion_matrix = calculate_confusion_matrix_for_each_model(llm_responces)

                #debug
                if DEBUG_LOG:
                    print(f"Cell datatype: {type(confusion_matrix[0][0])}")
                    for i in range(len(confusion_matrix)):
                        print(f"\t{i} {confusion_matrix[i]}")

                # append the discriptor for the dataframe
                if stat == 'dataframe':
                    stat = 'dataframe-stats'

                # lets plot it
                labels = [f"Interpretation {i+1}" for i in range(SAMPLE_COUNT)]
                title = f"Confusion Matrix: {sim_variable}, {model}, {stat}"
                plot_confusion_matrix(confusion_matrix, labels, title, output_file)

                # save confusion matrix to file both the plot and the data
                # cf_file_name = cm_path+f"/confusion_matrix_{sim_variable}_{model.replace(':','-')}_{stat}.json"
                confusion_matrix_dic = {
                    'matrix': confusion_matrix,
                    'labels': labels,
                    'title': title
                }

                #save confusion matrix data
                with open(cf_file_name, 'w', encoding="utf-8") as f:
                    json.dump(confusion_matrix_dic, f)

                # #lets try loading
                # with open(cf_file_name, 'r') as f:
                #     loaded_data = json.load(f)
                # loaded_confusion_matrix = loaded_data['matrix']
                # loaded_labels = loaded_data['labels']
                # loaded_title = loaded_data['title']

                # print("\n\nLoaded Confusion Matrix:\n", loaded_confusion_matrix)
                # print("Labels:", loaded_labels)
                # print("Title:", loaded_title)


def _get_n_interpretation_samples_of_statistical_results(sim_data_llm_results, sim_variable):
    ''' This method generates interpretations (n samples) of statistical results by each model '''

    sys_prompt = {
        'dataframe': 'You are an expert data statistician. You are analyzing results of Python dataframe.describe(), dataset’s summary statistics. Interpret the mean, standard deviation, min, 25%, 50%, 75%, and max values to understand the central tendency, dispersion, and distribution shape of the data.',
        'correlation': 'You are an expert data statistician. Given a correlation matrix, analyze the strength and direction of linear relationships between variables. Identify strongly correlated pairs (both positive and negative) and discuss implications for further analysis or modeling.',
        'linear-regression': 'You are an expert data statistician. Interpret the results of a simple linear regression, including the slope, intercept, and p-value. Discuss the significance of the relationship, the predictive power of the model, and any limitations or potential next steps in analysis.',
        'multicollinearity': 'You are an expert data statistician. Analyze the variance inflation factors (VIFs) to assess multicollinearity among variables. Identify variables with high VIF values, discuss the implications of multicollinearity on regression models, and suggest strategies for mitigating its effects.',
        'pca':'You are an expert data statistician. Given the results of a PCA, interpret the explained variance ratio of each principal component. Discuss how the data can be represented in a lower-dimensional space, identify the most contributing original variables to each component, and suggest potential applications or next steps.',
        'combined': 'You are an expert data statistician. Integrate the insights from summary statistics, correlation analysis, linear regression, multicollinearity assessment, and PCA. Synthesize these results to provide a comprehensive understanding of the dataset, including relationships between variables, potential issues with multicollinearity, and recommendations for data reduction or predictive modeling.'
    }

    # create a dictionary to save results
    llm_responces = {}
    for model in LLM_LIST:

        for stat in STATS_LIST:

            #define file names
            llm_input_prompt_file = intrp_path+f"/input_prompt_interpt_{sim_variable}_{model.replace(':','-')}_{stat}.txt"
            llm_interpreted_results = intrp_path+f"/llm_stat_results_interpretation_{sim_variable}_{model.replace(':','-')}_{stat}.json"

            # save statistical result input prompt to file
            llm_input_prompt = sim_data_llm_results[sim_variable][stat]['phi3:3.8b']['llm-input'].strip().replace('\n    ', '\n')
            with open(llm_input_prompt_file, "w", encoding="utf-8") as file:
                file.write(llm_input_prompt)

            #add model to dictionary
            llm_responces[model+stat] = []

            if os.path.exists(llm_interpreted_results):
                print(f"\n\tFile exists: {llm_interpreted_results} Skipping calculation.")

            else:
                if DEBUG_LOG:
                    print(f"  Sim Variable: {sim_variable}, stat={stat}\n\tsys_prompt: {sys_prompt[stat].strip()[0:MAX_OUTPUT_STRING_LENTH]}...\n\tllm_input_prompt: {llm_input_prompt.strip()[0:MAX_OUTPUT_STRING_LENTH]}...")

                for i in range(SAMPLE_COUNT):
                    r = _llm_call(system_prompt=sys_prompt[stat], input_text=llm_input_prompt, model=model)
                    llm_responces[model+stat].append(r)

                if DEBUG_LOG:
                    print(f"Model: {model}; len={len(llm_responces[model+stat])}")
                    for i, r in enumerate(llm_responces[model+stat]):
                        print(f"\t{i}: {r.strip()[0:MAX_OUTPUT_STRING_LENTH]}...")

                #add results to dictionary
                llm_interpretations = {
                    'sim-variable': sim_variable,
                    'stat-type': stat,
                    'model': model,
                    'interpretations': llm_responces
                }
               
                #save confusion matrix data
                with open(llm_interpreted_results, 'w', encoding="utf-8") as f:
                    json.dump(llm_interpretations, f)
    print("\nCompleted: _get_interpretation_of_statistical_results()")


def _calculate_semantic_similarity_for_interpreted_statistical_results(sim_data_llm_results, sim_variable):
    ''' This method calculates bert['f1'] similarity scores for each interpretation n times'''

    # create a dictionary to save results
    llm_responces = {}
    for model in LLM_LIST:

        #file name for storing sim results for each model
        file_sim_results = intrp_path+f"/sim_result_{model.replace(':','-')}.json"

        results = {}
        results['model'] = model
        results['pairwise_comparison_bertf1'] = {}
        results['sim_input_prompt_bertf1'] = {}
        for stat in STATS_LIST:

            #load llm input prompt
            llm_input_prompt = ""
            llm_input_prompt_file = intrp_path+f"/input_prompt_interpt_{sim_variable}_{model.replace(':','-')}_{stat}.txt"
            print(f"\n\tLoading: {llm_input_prompt_file}")
            with open(llm_input_prompt_file, 'r', encoding="utf-8") as f:
                llm_input_prompt = f.read().strip()
            
            if llm_input_prompt == "":
                print("\n\n\tLLM INPUT PROMPT NOT LOADED")

            #load llm results
            llm_interpretions = {}
            llm_interpretons_file = intrp_path+f"/llm_stat_results_interpretation_{sim_variable}_{model.replace(':','-')}_{stat}.json"
            with open(llm_interpretons_file, 'r', encoding="utf-8") as f:
                llm_interpretions = json.load(f)

            if llm_interpretions == {}:
                print("\n\n\tLLM INTERPRETATIONS NOT LOADED")
            
            #get interpretation list
            interpretion_list = llm_interpretions['interpretations'][model+stat]

            if DEBUG_LOG:
                print(f"\n\n\tLLM INPUT PROMPT: {llm_input_prompt[0:MAX_OUTPUT_STRING_LENTH]}...\n\tLLM INTERPRETATIONS (count = {len(interpretion_list)}: [0:{interpretion_list[0][0:MAX_OUTPUT_STRING_LENTH]}, ...]")

            # calculate bert similarity scores for pairwise comparison
            results['pairwise_comparison_bertf1'][stat] = pairwise_bert_similarity(interpretion_list)

            # calculate bert similarity scores with respect to the input prompt
            results['sim_input_prompt_bertf1'][stat] = calculate_bert_similarities_with_input(llm_input_prompt, interpretion_list)


        #save results to file
        with open(file_sim_results, 'w', encoding="utf-8") as f:
            json.dump(results, f)
            
        #display pairwise comparison results
        title = "Boxplot of pairwise similarity scores for: " + model
        file_plot_pairwise = intrp_path+f"/plot-pairwise-similarity-{model.replace(':','-')}.png"
        scores = []
        labels = []
        for key, value in results['pairwise_comparison_bertf1'].items():
            labels.append(key)
            scores.append(value)

        if DEBUG_LOG:
            print(f"\n\nDEBUG PAIRWISE COMPARISON SCORES\n\tLabels: {labels}\n\tScores: {scores}")
        
        #plot the results
        plot_pairwise_scores(scores, labels=labels, title=title, file_plot=file_plot_pairwise)

        #display similarity to input prompt
        title = "Boxplot for similarity scores with respect to the input prompt: " + model
        file_plot_input = intrp_path+f"/plot-similarity-to-input-{model.replace(':','-')}.png"
        scores = []
        labels = []
        for key, value in results['sim_input_prompt_bertf1'].items():
            labels.append(key)
            scores.append(value)

        if DEBUG_LOG:
            print(f"\n\nDEBUG COMPARISON TO INPUT SIMILAIRTY\n\tLabels: {labels}\n\tScores: {scores}")
        
        #plot the results
        plot_pairwise_scores(scores, labels=labels, title=title, file_plot=file_plot_input)
    print("\nCompleted: _get_interpretation_of_statistical_results()")
    

def _calculate_information_retention_averges(sim_data_llm_results, sim_variable):
    ''' This method calculates bert['f1'] similarity scores for each interpretation '''

    #file name for storing sim results for each model
    file_avg_results = intrp_path+f"/mean_semantic_similarity.ods"

    # create a dictionary to save results
    average_scores = {'model': []}
    for model in LLM_LIST:

        # model add model to results
        average_scores['model'].append(model)

        #read similarity results for each model
        model_interpretation_similarity_results = {}
        sim_results_file_path = intrp_path+f"/sim_result_{model.replace(':','-')}.json"
        print(f"\n\tLoading similarity results of LLM interpretations for {model} model, file: {sim_results_file_path}")
        with open(sim_results_file_path, 'r', encoding="utf-8") as f:
            model_interpretation_similarity_results = json.load(f)

        # # calculate bert similarity (F1) scores for pairwise comparison
        # for stat, score_lst in model_interpretation_similarity_results['pairwise_comparison_bertf1'].items():
        #     if stat not in average_scores:
        #         average_scores[stat] = []
        #     average_scores[stat].append(sum([item[2] for item in score_lst]) / len(score_lst) if score_lst else 0)

        # calculate bert similarty (F1)
        for stat, score_lst in model_interpretation_similarity_results['sim_input_prompt_bertf1'].items():
            if stat not in average_scores:
                average_scores[stat] = []
            average_scores[stat].append(sum([item[2] for item in score_lst]) / len(score_lst) if score_lst else 0)

    pretty_print_json(average_scores)

    # Write to ODS file
    df = pd.DataFrame(average_scores)
    print(f"Saving results to spreadsheet: {file_avg_results}")
    with pd.ExcelWriter(file_avg_results, engine='odf') as writer:
        df.to_excel(writer, sheet_name='Mean Semantic Similarity Scores', index=False)

def load_llm_output_logs():
    '''
    This method loads the llm generated results (JSON format) and evaluates the results metrics provided by evaluate python library (huggingface)
    '''
    logs_folder = 'logs'
    file_path = os.getcwd() + '/' + logs_folder + '/llm-output.json'
    # file_path = os.getcwd() + '/' + logs_folder + '/llm-output-small.json'
    with open(file_path, 'r') as f:
        llm_results = json.load(f)
    
    print(f"\nLoaded llm_results, file: {file_path}")

    # SIM_VARIABLE_LIST = ['Aircraft Quantity', 'wMOG capacity', 'Processing time', 'Poisson lambda', 'Max Duration']
    SIM_VARIABLE_LIST = ['Aircraft Quantity']

    for sim_var in SIM_VARIABLE_LIST:

        # # confusion matrix
        # _create_confusion_matrix_from_statistical_interpretation_results(sim_data_llm_results=llm_results, sim_variable=sim_var)

        # 
        _get_n_interpretation_samples_of_statistical_results(sim_data_llm_results=llm_results, sim_variable=sim_var)
        # _calculate_semantic_similarity_for_interpreted_statistical_results(sim_data_llm_results=llm_results, sim_variable=sim_var)
        # _calculate_information_retention_averges(sim_data_llm_results=llm_results, sim_variable=sim_var)

load_llm_output_logs()



# DEBUG
# #calculate confusion matrix
# model = 'phi3:3.8b'
# stat = 'dataframe'
# confusion_matrix = [[1.0, [0.9103384017944336], [0.8656302690505981], [0.8362557291984558], [0.8866589665412903]], [[0.9103384017944336], 1.0, [0.9063129425048828], [0.8642001748085022], [0.925069272518158]], [[0.8656302690505981], [0.9063129425048828], 1.0, [0.8766637444496155], [0.9194387793540955]], [[0.8362557291984558], [0.8642001748085022], [0.8766637444496155], 1.0, [0.8729729652404785]], [[0.8866589665412903], [0.925069272518158], [0.9194387793540955], [0.8729729652404785], 1.0]]

# #for debugging purposes lets print the matrix
# print(f"\n\nConfusiton Matrix:\n{confusion_matrix}")

# labels = [f"Interpretation {i+1}" for i in range(SAMPLE_COUNT)]
# title = f"{model}-{stat} Confusion Matrix"
# plot_confusion_matrix(confusion_matrix, labels, title)
