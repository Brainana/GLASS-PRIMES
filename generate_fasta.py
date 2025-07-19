import requests

# Your list of UniProt IDs
uniprot_ids = [
    "A2RU14", "O15273", "O15537", "O43653", "O43914", "O60880", "O75360", "O75838", "O95843",
    "P01037", "P01308", "P01764", "P02647", "P02810", "P03950", "P04118", "P05114", "P07305",
    "P07951", "P09417", "P10746", "P12272", "P15382", "P17936", "P19429", "P19957", "P21912",
    "P25189", "P26998", "P28069", "P29965", "P35548", "P40337", "P42771", "P46779", "P48201",
    "P54845", "P55000", "P57054", "P60201", "P61244", "P0CG47", "Q00604", "Q02577", "Q05066",
    "Q07021", "Q07627", "Q08648", "Q13323", "Q15170", "Q15526", "Q15543", "Q15672", "Q15744",
    "Q16594", "Q16637", "Q19429", "Q19431", "Q19458", "Q19480", "Q19483", "Q19485", "Q19487",
    "Q19488", "Q19492", "Q19493", "Q19496", "Q19497", "Q19498", "Q19499", "Q19500", "Q19502",
    "Q19503", "Q19504", "Q19505", "Q19506", "Q19507", "Q19508", "Q19509", "Q19510", "Q19511",
    "Q19512", "Q19513", "Q19514", "Q19515", "Q19516", "Q19517", "Q19518", "Q19519", "Q19520",
    "Q19521", "Q19522", "Q19523", "Q19524", "Q19525", "Q19526", "Q19527", "Q19528", "Q19529",
    "Q19530", "Q5EG05", "Q5JTJ3", "Q6ISU1", "Q75WM6", "Q7RTU4", "Q7Z2X4", "Q7Z3Z2", "Q8IV16",
    "Q8N100", "Q8N6I4", "Q8N726", "Q8TBE3", "Q8WVJ9", "Q8WYQ3", "Q99217", "Q99683", "Q99880",
    "Q9BTL4", "Q9BYR0", "Q9BYR4", "Q9H320", "Q9H3J6", "Q9H6K4", "Q9H902", "Q9HBH7", "Q9HC23",
    "Q9NQZ6", "Q9NRN9", "Q9NVV9", "Q9P0L0", "Q9UBU3", "Q9UHF0", "Q9Y4U1", "Q9Y6H6", "Q96A32",
    "Q96B36", "Q96FT9"
]
# UniProt API endpoint (batch FASTA download)
url = 'https://rest.uniprot.org/uniprotkb/stream'
params = {
    'format': 'fasta',
    'query': ' OR '.join(uniprot_ids)
}

response = requests.get(url, params=params)
response.raise_for_status()

# Save to FASTA file
with open('proteins.fasta', 'w') as f:
    f.write(response.text)

print("Downloaded FASTA saved to proteins.fasta")
