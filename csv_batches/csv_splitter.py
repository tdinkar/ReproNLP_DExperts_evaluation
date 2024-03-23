import csv
import os

# define paths for experiment results
absolute_path = os.path.abspath(__file__)
working_dir = os.path.dirname(absolute_path)
parent_dir = os.path.abspath(os.path.join(working_dir, os.pardir))

filename = os.path.join(working_dir, 'human_eval_toxicity.csv')
output_filename = os.path.join(working_dir, 'batches/human_eval_toxicity_with_unique_ids.csv')
batch_path =  os.path.join(working_dir, 'batches/')
batch_names =  os.path.join(working_dir, 'batches/batch_names.csv')

batch_size = 30
batch_name = 1
unique_sample_id = 0
unique_id_list = []
master_unique_sample_id = []
sample_id_number_for_batch = 4 # starts at 4 to correspomd to MSForm template
all_rows = []
master_file_rows = []
prompt_ids = []
headers = ['batch_name', 'prompt_ids', 'unique_id', 'link_MSForms'] # headers for master file

with open(filename) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for i, row in enumerate(reader):

        if i == 0:
            header_row = row
            header_row.insert(0,'sample_id')
            header_row.insert(0, 'unique_id')
            header_row = [row[0],row[1],row[2],row[3],row[5],row[9]]
        else:
            unique_sample_id += 1
            unique_id_list.append(unique_sample_id) # add unique id to list for originial csv
            master_unique_sample_id.append(unique_sample_id) # add unique id to list for master csv
            row.insert(0, sample_id_number_for_batch) # add sample id to batch coreesponsing to MSForms
            row.insert(0, unique_sample_id) # add unique id to batch
            sample_id_number_for_batch += 1
            all_rows.append([row[0],row[1],row[2],row[3],row[5],row[9]]) # only keep prompt and generated text
            
            # Keep track of prompt ids etc for batch size
            prompt_ids.append(row[2])

            if len(all_rows) == batch_size:
                with open(batch_path + f'batch_{batch_name}.csv', 'w+', newline='') as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(header_row)
                    writer.writerows(all_rows)
                master_file_rows.append([f'batch_{batch_name}.csv', prompt_ids, master_unique_sample_id, ''])
                all_rows = []  # Clear the list for the next batch
                master_unique_sample_id = [] # Clear the list for the next batch
                prompt_ids = []
                sample_id_number_for_batch = 4
                batch_name += 1

    # # Check if there are remaining rows (not a multiple of batch_size)
    # if all_rows:
    #     with open(batch_path + f'batch_{batch_name}.csv', 'w+', newline='') as csv_file:
    #         writer = csv.writer(csv_file)
    #         writer.writerow(header_row)
    #         writer.writerows(all_rows)

# Keep track of csv files
with open(batch_names, 'w+', newline='') as master_file:
    writer = csv.writer(master_file)
    writer.writerow(headers)    
    writer.writerows(master_file_rows)

# add unique sample ids back to original CSV
with open(filename, 'r') as input_file, open(output_filename, 'w', newline='') as output_file:
    # Create CSV reader and writer objects
    reader = csv.reader(input_file)
    writer = csv.writer(output_file)
    # Read the header and add the new column header
    header = next(reader)
    header.append('Unique_sample_ID')
    # Write the updated header to the new CSV file
    writer.writerow(header)
    # Iterate through rows and add the new column data
    for row, new_data in zip(reader, unique_id_list):
        row.append(new_data)
        writer.writerow(row)

