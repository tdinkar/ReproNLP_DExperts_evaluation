import pandas as pd
import argparse
import os
import glob
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadResults():
    def __init__(self, path_to_csv, sample_cols): 
        self.sample_cols = sample_cols
        self.csv_batches = glob.glob(path_to_csv+'/*.csv')

    def load_csv(self, csv_file, experiment, transpose = True):
        '''
        If transpose = False, returns data where each row represents 
        an annotator, and each column the sample. 
        '''
        logger.info('Loading file: \n%s', csv_file)
        # Get all columns to use
        if experiment == 'intra_annotator':
            self.sample_cols.insert(0, 7)
            df = pd.read_csv(csv_file, usecols = self.sample_cols)
            column_names = df.columns.tolist()
            df.rename(columns = {column_names[0]: 'ID', column_names[1]: '1'}, inplace=True)
            # String replacements
            df = self.str_replacements_for_sample(df, 1, 30)
            grouped = df.groupby('ID')
            grouped_dfs = [group.drop('ID', axis=1) for _, group in grouped]
            return grouped_dfs
        else:
            df = pd.read_csv(csv_file, usecols = self.sample_cols)
            # Rename other columns
            column_names = df.columns.tolist()
            df.rename(columns = {column_names[0]: '1'}, inplace=True)
            # String replacements
            df = self.str_replacements_for_sample(df, 0, 29)
            # Get correct data depending on the experiment
            logger.info('Fetching data for experiment: %s', experiment)
            # Transposing the data to merge with unique IDs, not used for kripps alpha
            if transpose:
                df = df.transpose()
            logger.info('Finished loading file')
            return df

    def str_replacements_for_sample(self, df, start_col_idx, end_col_idx):
        '''
        Get rid of weird non-breaking space inserted from MSForms
        https://stackoverflow.com/questions/10993612/how-to-remove-xa0-from-string-in-python    
        and other string substitutions.   
        '''
        label_mapping = {'C equally fluent': 'C'}
        # First change the C label
        df.iloc[:, start_col_idx:end_col_idx + 1] = df.iloc[:, start_col_idx:end_col_idx + 1].applymap\
            (lambda x: label_mapping[x] if x in label_mapping else x)
        # Remove \xa0
        df.iloc[:, start_col_idx:end_col_idx + 1] = df.iloc[:, start_col_idx:end_col_idx + 1].applymap\
            (lambda x: x.replace(u'\xa0', u'',) if isinstance(x, str) else x)
        # Remove blank spaces
        df.iloc[:, start_col_idx:end_col_idx + 1] = df.iloc[:, start_col_idx:end_col_idx + 1].applymap\
            (lambda x: x.replace(u' ', u'',) if isinstance(x, str) else x)
        return df

    def merge_batches_with_unique_id(self, experiment):
        '''
        Returns merged DF of batches with an assigned unique sample ID
        Rows: sample number
        Columns: Annotator or unique sample ID
        '''
        dataframes_dict = {}
        for file in self.csv_batches:
            df = self.load_csv(file, experiment, transpose=True)
            batch_no = self.get_batch_name(file)
            dataframes_dict[batch_no] = df 
        # Assign unique ID to each batch df (stored together in dict)
        dataframes_dict = self.set_id_to_batches(dataframes_dict)
        # Concatenate dfs together
        df_unique_id = pd.concat(dataframes_dict.values(), ignore_index=True, axis=0)
        return df_unique_id

    def set_id_to_batches(self, dataframes_dict):
        '''
        Adds a unique ID back to each batch
        '''
        # logger.info('Adding unique IDs back to batches: %s', dataframes_dict.keys())
        # Adds unique id back to batches and merges them
        for batch_no in dataframes_dict.keys():
            batch_df = dataframes_dict[batch_no]
            batch_df = self.set_batch_col_name(batch_df)
            # Assign back unique ID to batch
            unique_id_start = (batch_no*30) - 29
            batch_df['Unique_sample_ID'] = range(unique_id_start, unique_id_start + 30)
            dataframes_dict[batch_no] = batch_df
        return dataframes_dict

    def set_batch_col_name(self, df):
        # renaming annotator columns to match other batches
        return df.rename(columns={col: f"{'Annotator'}{i+1}" \
                                        for i, col in enumerate(df.columns)})

    def get_batch_name(self, file_name):
        '''
        Gets batch number from CSV
        '''
        pattern = r'\((\w+)\)'
        match = re.search(pattern, file_name)
        if match:
            match_1 = match.group(1)
            batch_no = int(match_1.lstrip('b'))
        else:
            logger.error(f'Cannot find batch number,\
                            check regex expression:=\'{pattern}\' with file name=\'{file_name}\'.')
        return batch_no

if __name__ == '__main__':
    absolute_path = os.path.abspath(__file__)
    working_dir = os.path.dirname(absolute_path)
    parser = argparse.ArgumentParser(description='Load annotation results')
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    parser.add_argument('-p',
                        '--path_to_csv',
                        type=str,
                        default=working_dir+'/repro_nlp_batch_B',
                        help='Folder name for the experiment results')
    
    parser.add_argument('-c',
                        '--sample_cols',
                        type=list,
                        default=list(range(9, 39)),
                        help='The columns process from the csv file.\
                            Only keeps the columns containing the samples from the batch.\n\
                            FORMAT NOTE: MUST GIVE an index range for\
                            the samples in the batch.\n \
                            Example: list(range(index first sample number, index last sample number))')
    
    args = parser.parse_args()
    data = LoadResults(args.path_to_csv, args.sample_cols)