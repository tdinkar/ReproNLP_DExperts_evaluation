from krippendorff_alpha import krippendorff_alpha, nominal_metric
from plot_test import plt_figure
from load_results import LoadResults
import pandas as pd
import os
import logging
import numpy as np


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalyseResults():
    def __init__(self, path_to_csv, experiment, sample_cols, key_csv): 
        logger.info('Checking results for experiment: %s', experiment)
        self.experiment = experiment
        self.data = LoadResults(path_to_csv, sample_cols)
        self.key_csv = pd.read_csv(key_csv)

    def calculate_percentages(self):
        '''
        Function to reproduce the results as they are presented in the original paper,
        i.e. using a percentage. 
        '''
        results_dict = {}
        logger.info('Calculating results for experiment: %s', self.experiment)
        # Fetch data with unique ids from load_results
        df_unique_id = self.data.merge_batches_with_unique_id(self.experiment)
        # Merge annotation results with original key csv
        df = self.merge_dfs_with_key(df_unique_id)
        # Replace annotator labels with preferred model output
        annotator_columns = df.filter(like='Annotator').columns
        for col in annotator_columns:
            df[col] = df.apply(lambda row: self.replace_values(row[col], row['sourcea'], row['sourceb']), axis=1)     
        # Group together rows based on sourcea+b model
        df['models'] = df.apply(lambda row: frozenset([row['sourcea'], row['sourceb']]), axis=1)
        grouped = df.groupby('models')
        grouped_dfs = [group.drop('models', axis=1) for _, group in grouped]
        # Calculate final results
        for sub_df in grouped_dfs:
            # filter out only annotator columns and concat
            all_labels = pd.concat(sub_df[column] for column in annotator_columns)
            label_percentages = all_labels.value_counts(normalize=True) * 100
            logger.info('The label percentages for experiment %s\n %s: %.2f vs. system %s: %.2f, %s: %.2f', 
            self.experiment, label_percentages.index[0], label_percentages.iloc[0].round(3),
            label_percentages.index[1], label_percentages.iloc[1].round(3),
            label_percentages.index[2], label_percentages.iloc[2].round(3))
            results_dict = self.format_results(label_percentages, results_dict)
        return results_dict
        # plt_figure(res=results_dict)

    def format_results(self, label_percentages, results_dict):
        # Format results to match input of original study
        if label_percentages.index[0] != 'DExperts':
            label_percentages = label_percentages[::-1]  # Reverse the order if 'DExperts' is not the first value
        # Construct the key name for the dictionary, excluding 'equal' if present
        key_name_parts = [label for label in label_percentages.index if label != 'equal']
        key_name = ','.join(key_name_parts)
        # Format results to match the input of the original study
        results_dict[key_name] = {self.experiment: label_percentages}
        return results_dict
        
    def merge_dfs_with_key(self, df_unique_id):
        # Merges the annotation results with the input prompts
        merged_df = self.key_csv.copy()
        merged_df = pd.merge(merged_df, df_unique_id, on='Unique_sample_ID', how='left')
        merged_df = merged_df.dropna()
        return merged_df
    
    def replace_values(self, value, sourcea, sourceb):
        '''
        Function to replace label values with model name
        '''
        if value == 'A':
            return sourcea
        elif value == 'B':
            return sourceb
        elif value == 'C':
            return 'equal'

    def kripps_alpha(self, write_latex = False):
        '''
        Calculates kripp's alpha for individual batches,
        writw_latex generates the results in latex format.
        '''
        csv_batches = self.data.csv_batches
        label_mapping = {'A': 1, 'B': 2, 'C': 3}

        if self.experiment == 'intra_annotator':
            intra_results = []
            for raw_batch in csv_batches:
                grouped_batch = self.data.load_csv(raw_batch, self.experiment, transpose = False)
                for group in grouped_batch:
                    batch = group.applymap(lambda x: label_mapping[x] if x in label_mapping else x)
                    intra_results.append(batch.to_numpy())
            combined_array = self.transform_data_for_intra(intra_results).to_numpy()
            logger.info(f'Calculating krippendorf\'s alpha for all batches for experiment: {self.experiment}')
            alpha = krippendorff_alpha(combined_array, nominal_metric,  missing_items="nan")
            logger.info("Annotator agreement for nominal metric: %.3f" \
                        % alpha)
        else:
            batch_name_list = []
            agreement_list = []
            for raw_batch in csv_batches:
                # Get batch name
                batch_name = self.data.get_batch_name(raw_batch)
                batch_name_list.append(batch_name)
                # Calculate annotator agreement
                batch = self.data.load_csv(raw_batch, self.experiment, transpose = False)
                batch = batch.applymap(lambda x: label_mapping[x] if x in label_mapping else x)
                batch_array = batch.to_numpy()
                logger.info(f'Calculating krippendorf\'s alpha for batch {batch_name} for experiment: {self.experiment}')
                alpha = krippendorff_alpha(batch_array, nominal_metric)
                logger.info("Annotator agreement for nominal metric: %.3f" \
                            % alpha)
                agreement_list.append(alpha)
        # write to latex
        if write_latex:
            self.write_to_latex(batch_name_list, agreement_list)

    def transform_data_for_intra(self, arrays):
        '''
        Creates an array grouping all annotators for intra-annotator agreement together, 
        to be used as input to the kripendorff_alpha function.
        Example output array for 2 annotators, where each row represents an annotator's 
        previous and new labels, and each column represents a sample:
        [ 1.  1. nan nan]
        [ 1.  1. nan nan]
        [nan nan 2.  1. ]
        [nan nan 3.  1. ]
        '''
        # Create a DataFrame from the first array
        df = pd.DataFrame(arrays[0])
        # Replace values in subsequent arrays with nan values
        for i in range(1, len(arrays)):
            temp_df = pd.DataFrame(arrays[i])
            temp_df.columns = [col + 30 * i for col in temp_df.columns]
            # Replace values with 'nan' based on column index
            for col in range(30):
                df[col + 30 * i] = np.nan
                temp_df[col] = np.nan
            df = pd.concat([df, temp_df], ignore_index=True, axis=0)
        return df
        
    def write_to_latex(self, batch_name_list, agreement_list):
        df = pd.DataFrame({'Batch no': batch_name_list, 'Kripp\'s alpha': agreement_list})
        df_sorted = df.sort_values(by='Kripp\'s alpha')
        print(df_sorted.to_latex(index=False, float_format="{:.3f}".format))

def experiment_wrapper():
    results_dict = {}
    # Get results from the original paper
    orig_results = {
        'DExperts,GPT-2': pd.Series({'DExperts': 30.0, 'GPT-2': 30.0, 'equal': 40.0}),
        'DExperts,DAPT': pd.Series({'DExperts': 26.0, 'DAPT': 35.0, 'equal': 39.0}),
        'DExperts,PPLM': pd.Series({'DExperts': 37.0, 'PPLM': 31.0, 'equal': 33.0}),
        'DExperts,GeDi': pd.Series({'DExperts': 36.0, 'GeDi': 28.0, 'equal': 35.0})
    }
    for key, value in orig_results.items():
        results_dict[key] = {'Original': value}

    # define paths for experiment results
    absolute_path = os.path.abspath(__file__)
    working_dir = os.path.dirname(absolute_path)
    parent_dir = os.path.abspath(os.path.join(working_dir, os.pardir))
    key_csv = os.path.join(parent_dir, 'csv_batches/batches/human_eval_toxicity_with_unique_ids.csv')

    # Experiments
    experiments = ['repro_nlp', 'higher_annotators', 'definition_fluency']
    
    # Iterate over experiments
    for i, experiment in enumerate(experiments, start=1):
        logger.info('---------------------------------------- Running Experiment: %s ----------------------------------------', experiment)
        path_to_csv = os.path.join(working_dir, experiment)
        # Analysis for the experiment
        results = AnalyseResults(path_to_csv, experiment, list(range(9, 39)), key_csv)
        logger.info('---------------------------------------- Krippendorf\'s alpha results ----------------------------------------')
        kripps_alpha = results.kripps_alpha()

        logger.info('---------------------------------------- Results reported as original study ----------------------------------------')
        experiment_results = results.calculate_percentages()
        for key, value in experiment_results.items():
            if key in results_dict:
                results_dict[key].update(value)
            else:
                results_dict[key] = value

    results_dict = convert_percentages_to_fractions(results_dict)
    plt_figure(results_dict, working_dir)

    logger.info('---------------------------------------- Running Experiment: %s ----------------------------------------', 'intra_annotator')
    path_to_csv = os.path.join(working_dir, 'intra_annotator')
    intra_results = AnalyseResults(path_to_csv, 'intra_annotator', list(range(9, 39)), key_csv)
    logger.info('---------------------------------------- Krippendorf\'s alpha results ----------------------------------------')
    kripps_alpha = intra_results.kripps_alpha()


def convert_percentages_to_fractions(data):
    for key, inner_dict in data.items():
        for sub_key, sub_dict in inner_dict.items():
            for sub_sub_key, value in sub_dict.items():
                if isinstance(value, float):
                    data[key][sub_key][sub_sub_key] = round(value / 100, 2)
    return data

if __name__ == '__main__':
    experiment_wrapper()
        


