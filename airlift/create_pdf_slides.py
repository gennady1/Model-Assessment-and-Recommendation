import os
import json
import ollama
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from grs_slide_generator import SlideGenerator

#--------DEFAULTS--------
SHOW_STAT_RESULTS        = False
PRINT_LLM_INPUT          = False
PRINT_LLM_OUTPUT_VERBOSE = False
PRINT_LLM_OUTPUT_SUMMARY = True

#output slide type. supported formats: pdf, html, powerpoint
SLIDE_TYPE = "pdf"

# LLM model
LLM_MODEL = 'gemma3:4b'               # Tiny model relatively speaking
# LLM_MODEL = 'mistral-nemo:12b'      # Good model, but 7.1 GB size.
# LLM_MODEL = 'llama3.3:latest'       # cannot output as JSON format... have to pass output to another LLM to generate 
# LLM_MODEL = 'deepseek-r1:70b'
# LLM_TEMPERATURE = 1.5 # very creative
# LLM_TEMPERATURE = 0.1 # 0: conservative (good for coding and correct syntax)
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.05

#create seperate folders for CSVs, and Images
csv_folder = "csv"          # note that csv data is generated in generate_scenario_csv_data.py
img_folder = "images"
img_path = os.path.join(os.getcwd(), img_folder)
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

#import slide generator
slide_generator = SlideGenerator(SLIDE_TYPE)


# --- Helper functions ---
def summarize_to_key_recommendations(analysis_type, llm_analysis):

    if PRINT_LLM_OUTPUT_VERBOSE:
        print(f"\nLOG (Verbose) Analysis Type = {analysis_type}:\n{llm_analysis}")

    llm_input = f"""
    I need you to identify top 2 key recommendations from {analysis_type} analysis of simulation data. The input variable '{TARGET_VARIABLE}' Here is the verbose analysis:
   
    
    {llm_analysis}

    The output must be in JSON format. Using this template for the output:
    {{
        "analysis_type": "{analysis_type}",
        "recommendation-1": "string",
        "recommendation-2": "string"
    }}

    Do not include JSON markers.
    """

    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_input
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )

    summarized_analysis = response['message']['content']

    if PRINT_LLM_OUTPUT_SUMMARY:
        print(f"\nLOG: Summarized {analysis_type} Analysis:\n{summarized_analysis}")

    return summarized_analysis


OUTPUT_SUMMARY_TXT_FILE = ""
for TARGET_VARIABLE in ['Aircraft Quantity', 'wMOG capacity', 'Processing time', 'Poisson lambda', 'Max Duration']:
    
    #load the proper CSV file
    if TARGET_VARIABLE == 'Aircraft Quantity':
        df = pd.read_csv(csv_folder+'/results-1_aircraft-quantity.csv')
        OUTPUT_SUMMARY_TXT_FILE = 'LLM_Recomendation1_'+TARGET_VARIABLE.replace(" ", "_") + ".txt"

    elif TARGET_VARIABLE == 'wMOG capacity':
        df = pd.read_csv(csv_folder+'/results-2_wMOG.csv')
        OUTPUT_SUMMARY_TXT_FILE = 'LLM_Recomendation2_'+TARGET_VARIABLE.replace(" ", "_") + ".txt"

    elif TARGET_VARIABLE == 'Processing time':
        df = pd.read_csv(csv_folder+'/results-3_processing-duration.csv')
        OUTPUT_SUMMARY_TXT_FILE = 'LLM_Recomendation3_'+TARGET_VARIABLE.replace(" ", "_") + ".txt"

    elif TARGET_VARIABLE == 'Poisson lambda':
        df = pd.read_csv(csv_folder+'/results-4_dynamic-route-poisson.csv')
        OUTPUT_SUMMARY_TXT_FILE = 'LLM_Recomendation4_'+TARGET_VARIABLE.replace(" ", "_") + ".txt"
        df = df.drop(['Min Duration', ], axis=1)        # drop the column

    elif TARGET_VARIABLE == 'Max Duration':
        TARGET_VARIABLE = 'Max Duration'
        df = pd.read_csv(csv_folder+'/results-5_dynamic-route-minmax.csv')
        OUTPUT_SUMMARY_TXT_FILE = '/LLM_Recomendation5_'+TARGET_VARIABLE.replace(" ", "_") + ".txt"
        df = df.drop(['Min Duration', ], axis=1)        # drop the column


    # identify columns with column with no variance, i.e., column with values where min() == max() 
    no_variance_cols = [col for col in df.columns if df[col].min() == df[col].max()]

    print(f"\nTARGET VARIABLE: '{TARGET_VARIABLE}'\nPREPROCESSING SIMULATION DATA (dropping columns where variance = 0):")
    # Drop any rows with missing data (if any)
    df.dropna(inplace=True)

    # Drop column where all columns with no variance
    for col in no_variance_cols:
        df = df.drop([col], axis=1)
        print(f"\tRemoved column: '{col}'")

    #create data labels
    data_labels = [TARGET_VARIABLE, 'Simulation time (sec)', 'Total flight distance', 'Total Lateness', 'Total waiting to process steps', 'Missed Deliveries', 'Score']


    #--- DATA ANALYSIS ---#
    # ###################################################################################################################################################################
    # #####           1. DATA FRAME ANALYSIS - basic stats
    # ###################################################################################################################################################################

    # Get basic statistics for each column
    sim_data_stats = df[data_labels].describe().to_json()

    # get basic stats and convert them into json format
    dataframe_stats = str(json.dumps(json.loads(sim_data_stats), indent=2))
    if SHOW_STAT_RESULTS:
        print('basic stats:', dataframe_stats)


    #Analysis by LLM
    # print(f"\n\tDEBUG: dependent_vars{data_labels[1:]}")
    llm_input_basic_stats_analysis = f"""As an expert data scientist, analyze the results of the dataframe.describe() function. The independent variable is: '{TARGET_VARIABLE}'. The dependent variables are: {', '.join(data_labels[1:])}. 
    Please ensure that observations are strictly based only the provided data. The key dependent output variable of the simulation is the 'Score'. The 'Score' variable captures the accumulation of penalties for late and missed deliveries. The 
    lower the score the better the performance of the schedueling algorithm. As such, the analysis should also include the interpretation of 'Score' variable.  

    These are the data stats from dataframe df.describe().to_json() function:
    {sim_data_stats}


    Final commands:
    Please identify insightful information.
    Cannot reduce processing time, it is static.
    Bottlenecks are often a cause of delays and missed deliveries. 
    Summarize the analysis and provide recommendation.
    Please format the output as JSON, and include the following attriubes:
    {{
        "variable": "{TARGET_VARIABLE}",
        "observation": "string",
        "recommendation": "string"
    }}.

    Do not include JSON markers.
    """
    if PRINT_LLM_INPUT:
        print(f"\nLLM input (linear regression):\n{llm_input_basic_stats_analysis}")
    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_input_basic_stats_analysis
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )

    llm_analysis_basic_stats = response['message']['content']
    final_recommendation_basic_stats = summarize_to_key_recommendations("dataframe stats analysis", llm_analysis_basic_stats)

    # print(f"\nDEBUG: datastats table ::\n{dataframe_stats}\n::")

    ## make slides using slide generator
    image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-exploratory-data-analysis.png".lower().replace(" ", "-")
    slide_generator.add_slide_from_algorithm_description("dataframe.describe()")
    slide_generator.add_table_slide("Python dataframe stats", dataframe_stats)
    slide_generator.add_recommendation_slide("dataframe.describe()", image_url_for_recommendation_slide, final_recommendation_basic_stats)


    # ###################################################################################################################################################################
    # #####           2. EXPLORATORY DATA ANALYSIS (EDA) - Visualization of the sim data
    # ###################################################################################################################################################################
    print("\nI.   EXPLORATORY DATA ANALYSIS (EDA) - Visualization of the sim data")
    #1. ### a) Check the distribution of 'Aircraft Quantity'. b) Visualize and analyze other variables' distributions and relationships with 'Aircraft Quantity'.
    sns.pairplot(df, vars=data_labels, hue=TARGET_VARIABLE, kind='reg')  # Kind: reg = regression, scatter = scatter plot
    plt.suptitle('Exploratory Data Analysis (EDA) - Visualization of the sim data', fontsize=16, fontweight='bold')
    plt.savefig(image_url_for_recommendation_slide, bbox_inches='tight')
    if SHOW_STAT_RESULTS:
        print(df.describe())
        plt.show()
    else:
        print("\tSkipping visualizations...")
    plt.close()



    # ###################################################################################################################################################################
    # #####           3. CORRELATION ANALYSIS
    # ###################################################################################################################################################################
    print("\nII.  CORRELATION ANALYSIS")
    # calculate correlations between variables
    correlation_matrix = df.iloc[1:, :].corr().round(2)       # Exclude the first row (target variable)
    cor_flight_distance_traveled = correlation_matrix.iloc[data_labels.index('Total flight distance'), 0]
    cor_lateness = correlation_matrix.iloc[data_labels.index('Total Lateness'), 0]
    cor_total_waiting_to_process_steps = correlation_matrix.iloc[data_labels.index('Total waiting to process steps'), 0]
    cor_missed_deliveries = correlation_matrix.iloc[data_labels.index('Missed Deliveries'), 0]
    cor_score = correlation_matrix.iloc[data_labels.index('Score'), 0]

    correlation_analysis = {
        "Target variable": TARGET_VARIABLE,
        "correlates with": {
            'Total flight distance': cor_flight_distance_traveled,
            'Total Lateness': cor_lateness,
            'Total waiting to process steps': cor_total_waiting_to_process_steps,
            'Missed Deliveries': cor_missed_deliveries,
            'Score': cor_score
        },
    }

    #Analysis by LLM
    llm_input_correlation_analysis = f"""As an expert data scientist, please analyze the results of the correlations analysis. These are the results of the dataframe.iloc[1:, :].corr() function:
    {correlation_analysis}

    Additional context:
    The independent input variable is: '{TARGET_VARIABLE}'. The dependent output variables are: {', '.join(data_labels[1:])}. The independent variable should predict the output variables.

    The key dependent output variable of the simulation is the 'Score'; it captures the performance of the schedueling algorithm. The 'Score' variable is the accumulation of penalties for late 
    and missed deliveries (lower is better). As such, the analysis should also include the interpretation of 'Score' variable. Please ensure that observations are based only on the provided data. 

    Final commands:
    Please identify insightful information.
    Summarize the analysis and provide recommendation.
    Please format the output as JSON, and include the following attriubes: 
    {{
        "variable": "{TARGET_VARIABLE}",
        "observation": "string",
        "recommendation": "string"
    }}.
    Do not include JSON markers.
    """

    if PRINT_LLM_INPUT:
        print(f"\nLLM input (liner regression):\n{llm_input_correlation_analysis}")
    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_input_correlation_analysis
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )
    llm_analysis_correlation = response['message']['content']

    # print("\n\nDEBUG-corr:\n", llm_analysis_correlation)
    final_recommendation_corr =  summarize_to_key_recommendations("correlation analysis", llm_analysis_correlation)

    image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-correlation-analysis.png".lower().replace(" ", "-")
    sns.heatmap(correlation_matrix, annot=True)
    plt.suptitle('Correlation Analysis', fontsize=16, fontweight='bold')
    plt.savefig(image_url_for_recommendation_slide, bbox_inches='tight')
    if SHOW_STAT_RESULTS:
        print("\nCorrelation_JSON:\n", json.dumps(correlation_analysis, indent=2))
        plt.show()
    plt.close()
        
    ## make slides using slide generator
    slide_generator.add_slide_from_algorithm_description("Correlations analysis: dataframe.iloc[].corr()")
    slide_generator.add_recommendation_slide("Correlations analysis", image_url_for_recommendation_slide, final_recommendation_corr)

    ###################################################################################################################################################################
    #####           4.  LINEAR REGRESSION ANALYSIS
    ###################################################################################################################################################################
    print("\nIII. LINEAR REGRESSION ANALYSIS")

    # define independent and dependent variables
    independent_var = TARGET_VARIABLE
    dependent_vars = []
    for col in df.columns.tolist():
        if col!= TARGET_VARIABLE:
            dependent_vars.append(col)

    print(f"\tIndependent variable: {independent_var}. Dependent Variables: {dependent_vars}")

    # Perform linear regression for each dependent variable
    regression_analysis_list = []
    for var in dependent_vars:
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[independent_var], df[var])
        regression_analysis = {
            "independent-variable": independent_var,
            "dependent-variable": var,
            "slope": slope,
            "intercept": intercept,
            "r-value": r_value,
            "p-value": p_value,
            "standard error": std_err
        }
        regression_analysis_list.append(regression_analysis)

    json_regression_analysis = {
        "analysis_type": "linear regression",
        "regression_analysis" : regression_analysis_list
    }

    #Analysis by LLM
    llm_input_ma_linear_regression_analysis = f"""As an expert data scientist, please analyze the results of the linear regression. These are the results of the scipy.stats.linregress() function:
    {json_regression_analysis}

    Additional context:
    The independent input variable is: '{TARGET_VARIABLE}'. The dependent output variables are: {', '.join(data_labels[1:])}.
    The "independent" variable should be able to predict "dependent" variable; in other words, the dependent variables are the consiquence (outcome) of the independent variable.

    The key dependent output variable of the simulation is the 'Score'; it captures the performance of the schedueling algorithm.
    The 'Score' variable is the accumulation of penalties for late and missed deliveries (lower is better).
    Cannot directly control simulation time. 
    As such, the analysis should also include the interpretation of 'Score' variable. Please ensure that observations are based only on the provided data. 

    Final commands:
    Please identify insightful information; summarize the analysis and provide recommendation. Please format the output as JSON, and include the following attriubes: 
    {{
        "variable": "{TARGET_VARIABLE}",
        "observation": "string",
        "recommendation": "string"
    }}.

    Do not include JSON markers.
    """
    if PRINT_LLM_INPUT:
        print(f"\nLLM input (linear regression):\n{llm_input_ma_linear_regression_analysis}")
    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_input_ma_linear_regression_analysis
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )
    llm_analysis_linear_regression = response['message']['content']
    final_recommendation_lin_reg =  summarize_to_key_recommendations("linear regression analysis", llm_analysis_linear_regression)

    if SHOW_STAT_RESULTS:
        print(json.dumps(json_regression_analysis,  indent=2))

    # Visualize the relationship between 'Aircraft Quantity' and each dependent variable
    for var in dependent_vars:
        image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-linear-regression-analysis-{var}.png".lower().replace(" ", "-")
        sns.regplot(x=independent_var, y=var, data=df)
        plt.suptitle('Linear Regression', fontsize=16, fontweight='bold')
        plt.title(f'{independent_var} vs {var}')
        plt.savefig(image_url_for_recommendation_slide, bbox_inches='tight')

        if SHOW_STAT_RESULTS:
            plt.show()
        plt.close()

    ## make slides using slide generator
    slide_generator.add_slide_from_algorithm_description("Linear regression analysis using Python's scipy.stats.linregress()")
    slide_generator.add_table_slide("Linear regression analysis", regression_analysis_list)
    slide_generator.add_recommendation_slide("Linear regression analysis", f"{img_folder}/{TARGET_VARIABLE}-linear-regression-analysis-total-lateness.png".lower().replace(" ", "-"), final_recommendation_lin_reg)


    ###################################################################################################################################################################
    #####           5. MULTICOLLINEARITY ANALYSIS
    ###################################################################################################################################################################
    print("\nIV.  MULTICOLLINEARITY ANALYSIS")

    #define X and y
    X = df[dependent_vars]
    y = df[independent_var]

    #Calculate Variance Inflation Factors (VIF) for each explanatory variable:
    vif = pd.DataFrame()
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif["features"] = X.columns

    #Analysis by LLM
    llm_input_vif_analysis = f"""As an expert data scientist, please analyze the results of the Variance Inflation Factors (VIF). These are the results of the statsmodels.stats.outliers_influence.variance_inflation_factor() function:
    {vif}

    Additional context:
    The 'Score' variable represents the overall performance of the schedueling algorithim within the simulation environment. It is the accumulation of penalties for late and missed deliveries; lower is better. 
    The key dependent output variable of the simulation is the 'Score'; it captures the performance of the schedueling algorithm. The 'Score' variable is the accumulation of penalties for late 
    and missed deliveries (lower is better). As such, the analysis should also include the interpretation of 'Score' variable. Please ensure that observations are based only on the provided data. 

    Final command:
    Please identify insightful information; summarize the analysis and provide recommendation. Please format the output as JSON, and include the following attriubes: 
    {{
        "variable": "{TARGET_VARIABLE}",
        "observation": "string",
        "recommendation": "string"
    }}.

    Do not include JSON markers.
    """
    if PRINT_LLM_INPUT:
        print(f"\nLLM input (Multicollinearity):\n{llm_input_vif_analysis}")

    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_input_vif_analysis
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )
    llm_analysis_vif_analysis = response['message']['content']
    final_recommendation_vif = summarize_to_key_recommendations("Variance Inflation Factors (VIF) analysis", llm_analysis_vif_analysis)

    # Visualize the relationship between 'Aircraft Quantity' and other variables
    for var in dependent_vars:
        image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-variance-inflation-factors-analysis-{var}.png".lower().replace(" ", "-")
        sns.scatterplot(x=independent_var, y=var, data=df)
        plt.xlabel(f'{independent_var} ({TARGET_VARIABLE})')
        plt.ylabel(f'{var}')
        plt.title(f'{independent_var} vs {var}')
        plt.suptitle('Variance Inflation Factors (VIF)', fontsize=16, fontweight='bold')
        plt.savefig(image_url_for_recommendation_slide, bbox_inches='tight')
        if SHOW_STAT_RESULTS:
            plt.show()
        plt.close()

    ## make slides using slide generator
    image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-variance-inflation-factors-analysis-total-waiting-to-process-steps.png".lower().replace(" ", "-")
    slide_generator.add_slide_from_algorithm_description("Variance Inflation Factors (VIF) - [Python] statsmodels.stats.outliers_influence.variance_inflation_factor()")
    slide_generator.add_recommendation_slide("Variance Inflation Factors (VIF) analysis", image_url_for_recommendation_slide, final_recommendation_vif)



    ###################################################################################################################################################################
    #####           6. PRINCIPAL COMPONENT ANALYSIS (PCA)
    ###################################################################################################################################################################
    print("\n\nV.   PRINCIPAL COMPONENT ANALYSIS:")
    # X, y already defined in the section II

    # Standardize features by removing the mean and scaling to unit variance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA with three principal components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)

    # Add the principal component scores back to the DataFrame
    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3'])
    df_pca[TARGET_VARIABLE] = df[TARGET_VARIABLE]

    # Explained variance ratios for each principal component
    explained_variance = pca.explained_variance_ratio_


    # Plot the principal component scores using a 3D scatter plot (scatterplot_matrix is from mpl_toolkits.mplot3d)
    image_url_for_recommendation_slide = f"{img_folder}/{TARGET_VARIABLE}-principal-component-analysis.png".lower().replace(" ", "-")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df_pca['PC1'], df_pca['PC2'], df_pca['PC3'], c=df_pca[TARGET_VARIABLE], cmap='viridis')
    plt.suptitle('Principal Component Analysis (PCA)', fontsize=16, fontweight='bold')
    plt.savefig(image_url_for_recommendation_slide, bbox_inches='tight')
    if SHOW_STAT_RESULTS:
        # Print the explained variance ratios for each principal component
        print(f"Explained variance by PC1, PC2, and PC3: {explained_variance}")
        # # Print the DataFrame with PCA scores and the target variable
        # print(df_pca)
        plt.show()
    plt.close()

    # #Analysis by LLM -- 3 Principalled components
    llm_pca_input = f"""As an expert data scientist, please analyze the results of the Principal Component Analysis (PCA) using the sklearn.decomposition.PCA() function.

    These are the results of the PCA with three principal components:
    Explained variance by PC1, PC2 and PC3: [{round(pca.explained_variance_ratio_[0], 4)}, {round(pca.explained_variance_ratio_[1], 4)}, {round(pca.explained_variance_ratio_[2], 4)}]

    PC1:
    - Loading for each variable:
    {[(var, round(pca.components_[0, i], 4)) for i, var in enumerate(X.columns.values)]}

    PC2:
    - Loading for each variable:
    {[(var, round(pca.components_[1, i], 4)) for i, var in enumerate(X.columns.values)]}

    PC3:
    - Loading for each variable:
    {[(var, round(pca.components_[2, i], 4)) for i, var in enumerate(X.columns.values)]}

    Can you please explain:
    1. What insights can we gain from these PCA results?
    2. How do the explained variances of PC1, PC2, and PC3 contribute to our understanding of the data structure?
    3. Can you interpret the loadings for each variable on all principal components?


    Additional context:
    The independent variable is: '{TARGET_VARIABLE}'. The dependent variables are: {', '.join(data_labels[1:])}.

    The key dependent output variable of the simulation is the 'Score'; it captures the performance of the schedueling algorithm. The 'Score' variable is the accumulation of penalties for late 
    and missed deliveries (lower is better). As such, the analysis should also include the interpretation of 'Score' variable. Please ensure that observations are based only on the provided data. 

    Final command:
    Please identify insightful information; summarize the analysis and provide recommendation. Please format the output as JSON, and include the following attriubes: 
    {{
        "variable": "{TARGET_VARIABLE}",
        "observation": "string",
        "recommendation": "string"
    }}.

    Do not include JSON markers.
    """
    if PRINT_LLM_INPUT:
        print(f"\nLLM input (PCA):\n{llm_pca_input}")
    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': llm_pca_input
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )
    llm_analysis_pca_analysis = response['message']['content']
    # llm_analysis_pca_analysis = strip_text_from_json(llm_analysis_pca_analysis)
    final_recommendation_pca = summarize_to_key_recommendations("principal component analysis", llm_analysis_pca_analysis)

    ## add to auto slide generator
    slide_generator.add_slide_from_algorithm_description("Principal Component Analysis (PCA) using Python's sklearn.decomposition.PCA()")
    slide_generator.add_recommendation_slide("Principal Component Analysis", image_url_for_recommendation_slide, final_recommendation_pca)


    ###################################################################################################################################################################
    #####           7. COMBINED ANALYSIS
    ###################################################################################################################################################################

    # llm_input_combined_analysis = f"""
    # {{
    #   "simulation-data-stats": {llm_analysis_basic_stats},
    #   "correlation_analysis": {llm_analysis_correlation},
    #   "linear-regression_analysis": {llm_analysis_linear_regression},
    #   "variance-inflation-factor_analysis": {llm_analysis_vif_analysis}
    #   "pca_analysis": {llm_analysis_pca_analysis}

    # }}
    # """

    llm_input_combined_analysis = f"""
    {{
    "simulation-data-stats": {final_recommendation_basic_stats},
    "correlation_analysis": {final_recommendation_corr},
    "linear-regression_analysis": {final_recommendation_lin_reg},
    "variance-inflation-factor_analysis": {final_recommendation_vif}
    "pca_analysis": {final_recommendation_pca}
    }}
    """

    # #make the input look nice
    # llm_input_combined_analysis = json.dumps(llm_input_combined_analysis, indent=2)

    #this template is used to summarize the results for the combined analysis
    llm_recomendation_template = f"""
    Please use the following answer template:
    {{
    "combined_analysis": {{
        "target_variable": "{TARGET_VARIABLE}",
        "simulation_data_stats": {{
            "observation": "string",
            "observation": "string",
        }},
        "correlation_analysis": {{
            "observation": "string",
            "observation": "string",
        }}
        "linear-regression_analysi": {{
            "observation": "string",
            "observation": "string",
        }}
        "variance-inflation-factor_analysis": {{
            "observation": "string",
        }}
        "multicollinearity analysis": {{
            "observation": "string",
        }}
        "recommendations": {{
            "recommendation": "string",
            "recommendation": "string"
        }}
    }}
    }}
    """


    #get the combined analysis results:
    combined_input = f"""
    Based on your knowledge of what you know about routing problems, and the analysis of the simulation data from the airlift challenge simulation environment.
    The analysis includes the following: basic data stats, correlation, regression, multicolllinearity and principal component analysis.
    The input target variable is: '{TARGET_VARIABLE}'. The attribute 'simulation-data-stats' provides the 
    statistical properties of the simulation data. The analysis of the simulation environment:
    {llm_input_combined_analysis}


    The key dependent output variable of the simulation is the 'Score'; it captures the performance of the schedueling algorithm.
    The 'Score' variable is the accumulation of penalties for late and missed deliveries (lower is better).
    As such, the analysis should also include the interpretation of 'Score' variable.
    Please ensure that interpretations are based only on the provided data. 

    Please identify insightful information, and generate a meta summary of the simulation results.
    Also, please identify any counter-intuitive behavior from data analysis,
    Use simple terms to interpret these results and proivde recommendations.

    Lastly, output must be in JSON format. Please use this template:
    {{
        "variable": "{TARGET_VARIABLE}",
        "observations": ["string", "string"],
        "recommendations": ["string", "string"],
        "counter-intuitive behavior":  "string",
    }}

    Do not include JSON markers.
    """

    if PRINT_LLM_INPUT:
        print(f"\n\n\n-------------------\nCombined LLM Input:\n", combined_input)
    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': combined_input
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0: very conservative (good for coding and correct syntax)
        }
    )
    final_analysis = response['message']['content']
    print(f"\n\n\nFinal Summary for variable = {TARGET_VARIABLE}:\n{final_analysis}")


    # llama3.3 does not generate JSON format, switch to mistral in that case
    if LLM_MODEL != 'mistral-nemo:12b':
        LLM_MODEL == 'mistral-nemo:12b'

    response = ollama.chat(model=LLM_MODEL, messages=[
            {
                'role': 'system',
                'content': 'You are an expert data analyst assistant.',     #1. set model context 
            },
            {
                'role': 'user',
                'content': f"""I have text that needs to be formated in JSON. PLease use this JSON template:
                {{
                    "variable": "{TARGET_VARIABLE}",
                    "observations": ["string", "string"],
                    "recommendations": ["string", "string"],
                    "counter-intuitive behavior":  "string",
                }}

                This is the text:
                {final_analysis}
                """
            },
        ],
        options = {
            # 'logit_bias': LLM_LOGIT_BIAS,
            'temperature': LLM_TEMPERATURE # 0.1: very conservative (good for coding and correct syntax)
        }
    )
    final_analysis = response['message']['content']

    slide_generator.add_recommendation_slide("Summary: Combined Analysis", f"airlift_challenge_logo.jpg", final_analysis)

    #strip non JSON data.
    start_index = final_analysis.find('```json')
    end_index = final_analysis.rfind('```')
    if start_index != -1 and end_index!= -1:
        final_analysis = final_analysis[start_index:end_index+3]


    print("\n\nConverted to JSON:\n", final_analysis)
        
    #save results_to file:
    with open(SLIDE_TYPE + "/"+OUTPUT_SUMMARY_TXT_FILE, 'w') as f:
        f.write(final_analysis)

    #create pdf slides
    slide_generator.save_marp_to_file(OUTPUT_SUMMARY_TXT_FILE)