import os
import pandas as pd
import numpy as np
import re
import ast


def parse_data_to_dict(data):
    lines = data.strip().split('\n')
    parsed_dict = {}

    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Attempt to convert value to a more appropriate type
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass  # Keep the value as a string if it can't be evaluated
            
            parsed_dict[key] = value
    
    return parsed_dict


def file_contents(folder_path, filename):
    
    data = []
    bodyHeadings = ['Chromosome', 'Hyperparameter Configuration', 'Fitness Score']
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            with open(f'{root}/{file}', 'r') as file:
                dictData = parse_data_to_dict(file.read())
                data.append(dictData)
                
    with pd.ExcelWriter(f'{filename}.xlsx', engine='openpyxl') as writer:
        for element in data:
            if 'Generation Data' in element:
                genData = element.pop('Generation Data')

            genData = [inner_list[:3] for inner_list in genData]
            
            bodySubHeadings = list(genData[0][1].keys()) # Subheadings for the hypeparameters
            
            for i in range(len(bodySubHeadings) - 1):
                if element['Generation Count'] != 1:
                    break
                bodyHeadings.insert(2, 'Hyperparameter Configuration');
            
            for index in range(len(genData)):
                i = 0
                hyperparameter = genData[index][1]
                genData[index].pop(1)
                for key in bodySubHeadings:
                    
                    value = hyperparameter[key]
                    genData[index].insert(1 + i, value)
                    i += 1
            
            bodySubHeadings.insert(0, '')
            bodySubHeadings.append('')
            
            multiIndex = pd.MultiIndex.from_arrays([bodyHeadings, bodySubHeadings])
            
            del element['Best Hyperparameter Configuration']
            headerDataFrame = pd.DataFrame(data=element, index=['Values'])
            bodyDataFrame = pd.DataFrame(data=genData, columns=multiIndex)
            
            headerDataFrame.to_excel(writer, sheet_name=f"Sheet_{element['Generation Count']}", startrow=0, index=True)
            bodyDataFrame.to_excel(writer, sheet_name=f"Sheet_{element['Generation Count']}", startrow=len(headerDataFrame) + 2)
            

def main():
    file_contents('Iteration_13', 'Iteration_13_report');

if __name__ == "__main__":
    main()