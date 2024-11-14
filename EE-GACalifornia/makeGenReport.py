import os
import pandas as pd
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

def process_generation_data(gen_data):
    """
    Process Generation Data to split hyperparameters and fitness scores
    """
    processed_data = []
    for entry in gen_data:
        if len(entry) < 2:
            print("Skipping entry due to incorrect format:", entry)
            continue  # Skip if the format is incorrect

        try:
            chromosome, hyperparameters = entry
            row = [chromosome]  # Start with chromosome
            row.extend(hyperparameters.values())  # Add hyperparameters
            processed_data.append(row)
        except Exception as e:
            print(f"Error processing entry {entry}: {e}")
    
    return processed_data
def file_contents(folder_path, filename):
    data = []
    
    # Read and parse files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                dictData = parse_data_to_dict(f.read())
                data.append(dictData)
    
    # Debug print
    print("Parsed Data:", data)
    
    if not data:
        print("No data found. Creating dummy sheet.")  # Debug print
        with pd.ExcelWriter(f'{filename}.xlsx', engine='openpyxl') as writer:
            pd.DataFrame({"Dummy": []}).to_excel(writer, sheet_name="DummySheet")
        return

    # Create Excel report
    with pd.ExcelWriter(f'{filename}.xlsx', engine='openpyxl') as writer:
        for element in data:
            if 'Generation Data' not in element:
                print("Skipping element with no Generation Data:", element)
                continue

            genData = element.pop('Generation Data')
            processed_gen_data = process_generation_data(genData)
            
            if not processed_gen_data:
                print("Skipping element with empty processed generation data:", element)
                continue

            # Set up headings
            bodySubHeadings = list(genData[0][1].keys())
            bodyHeadings = ['Chromosome'] + bodySubHeadings + ['Fitness Score']

            multiIndex = pd.MultiIndex.from_arrays([bodyHeadings])

            # Prepare DataFrames
            headerDataFrame = pd.DataFrame(data={
                'Model': [element.get('Model')],
                'Fitness Function': [element.get('Fitness Function')],
                'Objective': [element.get('Objective')],
                'Generation Count': [element.get('Generation Count')],
                'Highest Fitness': [element.get('Highest Fitness')],
                'Lowest Fitness': [element.get('Lowest Fitness')],
                'Average Fitness': [element.get('Average Fitness')],
                'Best Chromosome': [element.get('Best Chromosome')],
                'Best Hyperparameter Configuration': [element.get('Best Hyperparameter Configuration')]
            })
            
            bodyDataFrame = pd.DataFrame(data=processed_gen_data, columns=bodyHeadings)

            sheet_name = f"Sheet_{element.get('Generation Count', 1)}"
            print(f"Writing to sheet: {sheet_name}")  # Debug print
            headerDataFrame.to_excel(writer, sheet_name=sheet_name, startrow=0, index=True)
            bodyDataFrame.to_excel(writer, sheet_name=sheet_name, startrow=len(headerDataFrame) + 2, index=False)

def main():
    file_contents('Iteration_1', 'Iteration_1_report')

if __name__ == "__main__":
    main()