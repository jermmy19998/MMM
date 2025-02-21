import pandas as pd

df = pd.read_csv('./tcga_luad_all_clean.csv',low_memory=False)

slide_ids = df['slide_id'].tolist()

file_path = './luad.txt'

valid_lines = []

with open(file_path, 'r') as file:
    header = file.readline().strip()
    
    for line in file:
        line = line.strip()
        
        columns = line.split('\t')
        
        if len(columns) > 1:
            filename = columns[1]
            
            if filename.endswith('.svs') and filename in slide_ids:
                valid_lines.append(line)

output_path = '/clean_luad.txt'
with open(output_path, 'w') as output_file:
    output_file.write(header + '\n')
    
    for valid_line in valid_lines:
        output_file.write(valid_line + '\n')
