#%%
import csv

def tsv_to_csv(tsv_file, csv_file):
    with open(tsv_file, 'r', newline='', encoding='utf-8') as tsv_in:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csv_out:
            tsv_reader = csv.reader(tsv_in, delimiter='\t')
            csv_writer = csv.writer(csv_out, delimiter=',')
            for row in tsv_reader:
                csv_writer.writerow(row)

tsv_to_csv('/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/marketing_campaign.tsv', '/Users/peggy/Documents/uu_master_data_science/uu_data_intensive_systems_group_project/Data/marketing_campaign_converted.csv')