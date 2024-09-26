import requests
from bs4 import BeautifulSoup
import csv
import time

def scrape_uniprot_accession(pdb_id):
    url = f"https://www.rcsb.org/structure/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        uniprot_link = soup.find('a', href=lambda x: x and 'uniprot.org/uniprot' in x)
        if uniprot_link:
            uniprot_accession = uniprot_link['href'].split('/')[-1]
            return uniprot_accession
        else:
            print("UniProt accession number not found.")
            return None
    else:
        print(f"Failed to retrieve data for PDB ID {pdb_id}: {response.status_code}")
        return None

def get_pdbs(file_type):
    id_list = []
    with open(f'scratch/protein-datasets/GeneOntology/nrPDB-GO_{file_type}.txt', 'r') as file:
        for line in file:
            pdb_id = line.split('-')[0]
            id_list.append(pdb_id.strip())
    return id_list

train_pdb_ids = get_pdbs('train')

with open('scratch/protein-datasets/GeneOntology/train_pdb2uniprot.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['PDB ID', 'UniProt Accession'])

    for idx, pdb_id in enumerate(train_pdb_ids):
            uniprot_acc = scrape_uniprot_accession(pdb_id)
            csvwriter.writerow([pdb_id, uniprot_acc])
            time.sleep(0.1)
            if idx%1000 == 0:
                  print('checkpoint: ', idx)

