import os
import re
import json

def extract_mu_ga(filename):
    # 使用正则表达式提取mu和ga的值
    match = re.search(r'gen_data_mu(\d+(\.\d+)?e?-?\d*)_ga(\d+(\.\d+)?e?-?\d*)', filename)
    if match:
        mu = match.group(1)
        ga = match.group(3)
        return mu, ga
    return None, None

def organize_files_by_mu_ga(directory):
    mu_ga_dict = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.h5'):
                mu, ga = extract_mu_ga(file)
                if mu and ga:
                    if mu not in mu_ga_dict:
                        mu_ga_dict[mu] = {}
                    if ga not in mu_ga_dict[mu]:
                        mu_ga_dict[mu][ga] = []
                    relative_path = os.path.join(root, file)
                    mu_ga_dict[mu][ga].append(relative_path)
    return mu_ga_dict

def save_dict_to_json(data, output_file):
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def main():
    directory = './data_diffusion/diffusion/'
    output_file = 'mu_ga_files_new.json'
    mu_ga_dict = organize_files_by_mu_ga(directory)
    save_dict_to_json(mu_ga_dict, output_file)
    print(f'Dictionary saved to {output_file}')

if __name__ == '__main__':
    main()
