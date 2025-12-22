''' This class evaluates semantic similarity between the stats and the LLM interetation the stats. 
    This is done using Huggingface evaluate library (specifically the  BERT embadding)
    This method is called by:  metrics_evaluate_llm_interpretation( llm_results )

    UPDATES:
    20251118 - Added descriptions.
'''

import os
import json
import concurrent.futures

import evaluate
import pandas as pd

# --------------------------------------------------------
# These are helper functions for hugging face evaluate 
# method calls (in paralle)
# --------------------------------------------------------

def _compute_bleu(predictions, reference):
    return evaluate.load("bleu").compute(predictions=predictions, references=reference)['bleu']

def _compute_rouge(predictions, reference):
    try:
        r = evaluate.load("rouge").compute(predictions=predictions, references=[reference])
        return {'rouge1': float(r['rouge1']), 'rouge2' : float(r['rouge2']), 'rougeL' : float(r['rougeL']), 'rougeLsum' : float(r['rougeLsum'])}
    except Exception as e:
        print(f"\n\n\t** Exception-ROUGE: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0 }

def _compute_meteor(predictions, reference):       # <-- tell Python to use the global variable

    return float(evaluate.load('meteor').compute(predictions=predictions, references=reference)['meteor'])

def _compute_bertscore(predictions, reference):
    try:
        r = evaluate.load('bertscore', model_type="roberta-large").compute(lang="en", predictions=predictions, references=reference)
        return {'precision': r['precision'][0], 'recall': r['recall'][0], 'f1': r['f1'][0]}
    except Exception as e:
        print(f"\n\t** Exception-BERTScore: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

def _evaluate_interpretation(target_variable, data_process_pipeline, pipeline_llm_results):
    ''' This method evaluates how much semantic information is retained between the stats and the LLM interetation using Huggingface evaluate library (using BERT embadding)
        This method is called by:  metrics_evaluate_llm_interpretation( llm_results )
    '''

    print(f"\n\n  DATA_PROCESS_PIPELINE: {data_process_pipeline}\n")

    # initialize lists for tracking metrics interpretation scores
    model_lst = []
    interpretation_lists = {
        'bertscore_precision': [],
        'bertscore_recall': [],
        'bertscore_f1': []
    }

    summary_lists = {
        'rougeLsum': [],
        'meteor': [],
        'bertscore_f1': []
    }

    best_interpretation = ""
    best_interpretation_model = ""
    best_interpretation_score_bertscore = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    best_summary = ""
    best_summary_model = "" 
    best_summary_score_rouge_lsum = 0.0
    best_summary_score_meteor = 0.0
    best_summary_score_bertscore_f1 = 0.0


    #iterate over llm data
    for model, values in pipeline_llm_results.items():
        # print(f"\n\nDEBUG: model_process_task={model} values.keys()={values.keys()}")

        #keep track of model information - this will be used for dataframe column with model names
        model_lst.append(model)

        input_text = [values['llm-input']]
        interpretation_of_results = [values['interpretation']]
        summarized_interpretation = [values['summary']]

        # Call evaluation functions concurrently, hopefully this will speed things up a bit
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_compute_bertscore, interpretation_of_results, input_text): 'interpret-bertscore',
                executor.submit(_compute_rouge, summarized_interpretation, interpretation_of_results): 'summary-rouge',
                executor.submit(_compute_meteor, summarized_interpretation, interpretation_of_results): 'summary-meteor',
                executor.submit(_compute_bertscore, summarized_interpretation, interpretation_of_results): 'summary-bertscore',
            }

            for future in concurrent.futures.as_completed(futures):
                metric_type = futures[future]
                result = future.result()

                if metric_type == 'interpret-bertscore':
                    
                    # recocord interpretation results            
                    interpretation_lists['bertscore_precision'].append(result['precision'])
                    interpretation_lists['bertscore_recall'].append(result['recall'])
                    interpretation_lists['bertscore_f1'].append(result['f1'])

                    if ( result['precision'] > best_interpretation_score_bertscore['precision']
                      or result['recall'] > best_interpretation_score_bertscore['recall']
                      or result['f1'] > best_interpretation_score_bertscore['f1']):
                        best_interpretation_score_bertscore['precision'] = result['precision']
                        best_interpretation_score_bertscore['recall'] = result['recall']
                        best_interpretation_score_bertscore['f1'] = result['f1']
                        best_interpretation = interpretation_of_results
                        best_interpretation_model = model
       
                elif metric_type == 'summary-rouge':
                    summary_lists['rougeLsum'].append(result['rougeLsum'])      # record summary results

                    if (result['rougeLsum'] > best_summary_score_rouge_lsum):
                        best_summary_score_rouge_lsum = result['rougeLsum']
                        best_summary = summarized_interpretation
                        best_summary_model = model

                elif metric_type == 'summary-meteor':
                    summary_lists['meteor'].append(result)                      # record summary results

                    if result > best_summary_score_meteor:
                        best_summary_score_meteor = result
                        best_summary = summarized_interpretation
                        best_summary_model = model

                elif metric_type == 'summary-bertscore':
                    summary_lists['bertscore_f1'].append(result['f1'])          # record summary results

                    if (result['f1'] > best_summary_score_bertscore_f1):
                        best_summary_score_bertscore_f1 = result['f1']
                        best_summary = summarized_interpretation
                        best_summary_model = model

    # print(f"\n\n---DEBUG: interpretation_lists: {interpretation_lists}")
    # print(f"\n\n---DEBUG: summary_lists: {summary_lists}")
    # log_metrics_interpretation += f"\nDEBUG: [interpretation] model-{model} bert-precision={interpretation_lists['bertscore_precision'][-1]}, bert-recall={interpretation_lists['bertscore_recall'][-1]}, bert-f1={interpretation_lists['bertscore_f1'][-1]}"
    # log_metrics_summary += f"\nDEBUG: [summary] model-{model}: 'rougeLsum'={summary_lists['rougeLsum'][-1]}; 'meteor'={summary_lists['meteor'][-1]}; 'rougeLsum'={summary_lists['bertscore_f1'][-1]}"

    # create dataframes
    df_interpretation = pd.DataFrame({'model': model_lst})
    for list_name, list_value in interpretation_lists.items():
        df_interpretation[list_name] = list_value

    df_summary = pd.DataFrame({'model': model_lst})
    for list_name, list_value in summary_lists.items():
        df_summary[list_name] = list_value

    info_label_interp_results = pd.DataFrame(['Result interpretation']).T
    info_row_summary_results =  pd.DataFrame(['Summarization of results']).T
    separator = pd.DataFrame([''] * len(df_interpretation.columns)).T

    # Combine the DataFrames
    combined_df = pd.concat([
        info_label_interp_results,
        df_interpretation,
        # separator,
        separator,
        info_row_summary_results,
        df_summary
    ], ignore_index=True)

    # package the results
    results = {
        'combined_dataframe': combined_df,
        'best_interpretation_model' : best_interpretation_model,
        'best_interpretation' : best_interpretation,
        'best_summary' : best_summary,
        'best_summary_model' : best_summary_model
    }
    
    #return best summary
    return results

def debug_evaluate_llm_interpretation(llm_results, sim_variable='Aircraft Quantity'):
    print(f"\n\nEvaluating LLM responces for simulation-variable: {sim_variable}")

    PROCESSING_PIPELINE_LIST = ['combined', 'dataframe', 'correlation', 'linear-regression', 'multicollinearity', 'pca']

    # #iterate over each data processing pipleline
    sim_var = llm_results[sim_variable]
    if sim_var != None:
        target_results_df = {}
        for data_process_pipeline in PROCESSING_PIPELINE_LIST:
            pipeline_llm_results = sim_var[data_process_pipeline]
            target_results_df[data_process_pipeline] = _evaluate_interpretation(sim_variable, data_process_pipeline, pipeline_llm_results)['combined_dataframe']

        #write results to open document sheet
        print("Saving results to disk")
        with pd.ExcelWriter('./metrics/Interpretation_Summary_'+sim_variable.replace(' ', '_') + '.ods', engine='odf') as writer:
            for analysis_name, df in target_results_df.items():
                df.to_excel(writer, sheet_name=analysis_name)

def load_llm_output_logs_evaluate_llm_interpretation():
    '''
    This method loads the llm generated results (JSON format) and evaluates the results metrics provided by evaluate python library (huggingface)
    '''
    llm_results = {}
    logs_folder = 'logs'
    file_path = os.getcwd() + '/' + logs_folder + '/llm-output.json'
    # file_path = os.getcwd() + '/' + logs_folder + '/llm-output-small.json'
    with open(file_path, 'r') as f:
        llm_results = json.load(f)
    
    print(f"\nLoaded llm_results, file: {file_path}")

    # SIM_VARIABLE_LIST = ['Aircraft Quantity', 'wMOG capacity', 'Processing time', 'Poisson lambda', 'Max Duration']
    SIM_VARIABLE_LIST = ['Processing time', 'Poisson lambda', 'Max Duration']

    for sim_var in SIM_VARIABLE_LIST:
        debug_evaluate_llm_interpretation(llm_results=llm_results, sim_variable=sim_var)

    
load_llm_output_logs_evaluate_llm_interpretation()
# test_metrics()
