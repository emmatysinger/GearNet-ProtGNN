import requests
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

def uniprot_accession_to_gene_name(accession):
    if not accession or accession == np.NaN:
        return None
    
    url = f"https://www.uniprot.org/uniprot/{accession}.xml"
    response = requests.get(url)
    
    if response.status_code == 200:
        
        root = ET.fromstring(response.content)
        for entry in root.findall('{http://uniprot.org/uniprot}entry'):
            for gene in entry.findall('{http://uniprot.org/uniprot}gene'):
                for name in gene.findall('{http://uniprot.org/uniprot}name'):
                    if name.attrib['type'] == 'primary':
                        return name.text
    else:
        print(f"Failed to retrieve data for UniProt Accession {accession}: {response.status_code}")
        return None


input_csv = 'scratch/protein-datasets/GeneOntology/train_pdb2uniprot.csv'
output_csv = 'scratch/protein-datasets/GeneOntology/train_pdb2uniprot2genename.csv'

df = pd.read_csv(input_csv)
df['Gene Name'] = df['UniProt Accession'].apply(uniprot_accession_to_gene_name)
df.to_csv(output_csv)
