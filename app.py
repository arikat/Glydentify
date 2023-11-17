import pandas as pd
from IPython.display import clear_output
import torch
from transformers import EsmForSequenceClassification, AdamW, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import torch.nn.functional as F
import gradio as gr
import io
from PIL import Image
import Bio
from Bio import SeqIO
import zipfile
import os

# Load the model from the file
with open('family_labels.pkl', 'rb') as filefam:
    yfam = pickle.load(filefam)

tokenizerfam = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D") #facebook/esm2_t33_650M_UR50D

device = 'cpu'
device

modelfam = EsmForSequenceClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=len(yfam.classes_))
modelfam = modelfam.to('cpu')

modelfam.load_state_dict(torch.load("family.pth", map_location=torch.device('cpu')))
modelfam.eval()

x_testfam = ["""MAEVLRTLAGKPKCHALRPMILFLIMLVLVLFGYGVLSPRSLMPGSLERGFCMAVREPDH
LQRVSLPRMVYPQPKVLTPCRKDVLVVTPWLAPIVWEGTFNIDILNEQFRLQNTTIGLTV
FAIKKYVAFLKLFLETAEKHFMVGHRVHYYVFTDQPAAVPRVTLGTGRQLSVLEVRAYKR
WQDVSMRRMEMISDFCERRFLSEVDYLVCVDVDMEFRDHVGVEILTPLFGTLHPGFYGSS
REAFTYERRPQSQAYIPKDEGDFYYLGGFFGGSVQEVQRLTRACHQAMMVDQANGIEAVW
HDESHLNKYLLRHKPTKVLSPEYLWDQQLLGWPAVLRKLRFTAVPKNHQAVRNP
"""]

encoded_inputfam = tokenizerfam(x_testfam, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_idsfam = encoded_inputfam["input_ids"]
attention_maskfam = encoded_inputfam["attention_mask"]

with torch.no_grad():
    outputfam = modelfam(input_idsfam, attention_mask=attention_maskfam)
    logitsfam = outputfam.logits
    probabilitiesfam = F.softmax(logitsfam, dim=1)
    _, predicted_labelsfam = torch.max(logitsfam, dim=1)
probabilitiesfam[0]

decoded_labelsfam = yfam.inverse_transform(predicted_labelsfam.tolist())
decoded_labelsfam



#Load donor model from file
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")

with open('donor_labels.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# encoded_labels = label_encoder.fit(y)
# labels = torch.tensor(encoded_labels)

model = EsmForSequenceClassification.from_pretrained("facebook/esm2_t12_35M_UR50D", num_labels=len(label_encoder.classes_))
model = model.to('cpu')

model.load_state_dict(torch.load("best_model_35M_t12_5v5.pth", map_location=torch.device('cpu'))) #model_best_35v2M.pth
model.eval()

x_test = ["""MAEVLRTLAGKPKCHALRPMILFLIMLVLVLFGYGVLSPRSLMPGSLERGFCMAVREPDH
LQRVSLPRMVYPQPKVLTPCRKDVLVVTPWLAPIVWEGTFNIDILNEQFRLQNTTIGLTV
FAIKKYVAFLKLFLETAEKHFMVGHRVHYYVFTDQPAAVPRVTLGTGRQLSVLEVRAYKR
WQDVSMRRMEMISDFCERRFLSEVDYLVCVDVDMEFRDHVGVEILTPLFGTLHPGFYGSS
REAFTYERRPQSQAYIPKDEGDFYYLGGFFGGSVQEVQRLTRACHQAMMVDQANGIEAVW
HDESHLNKYLLRHKPTKVLSPEYLWDQQLLGWPAVLRKLRFTAVPKNHQAVRNP
"""]

encoded_input = tokenizer(x_test, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = encoded_input["input_ids"]
attention_mask = encoded_input["attention_mask"]

with torch.no_grad():
    output = model(input_ids, attention_mask=attention_mask)
    logits = output.logits
    probabilities = F.softmax(logits, dim=1)
    _, predicted_labels = torch.max(logits, dim=1)
probabilities[0]

decoded_labels = label_encoder.inverse_transform(predicted_labels.tolist())
decoded_labels


glycosyltransferase_db = {
    "GT31-chsy"      : {'CAZy Name': 'GT31', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT31.html'},
    "GT2-CesA2"      : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT43-arath"     : {'CAZy Name': 'GT43', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT43.html'},
    "GT8-Met1"       : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT32-higher"    : {'CAZy Name': 'GT32', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT32.html'},
    "GT40"           : {'CAZy Name': 'GT40', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT40.html'},
    "GT16"           : {'CAZy Name': 'GT16', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '6  ', 'More Info': 'http://www.cazy.org/GT16.html'},
    "GT27"           : {'CAZy Name': 'GT27', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '5  ', 'More Info': 'http://www.cazy.org/GT27.html'},
    "GT55"           : {'CAZy Name': 'GT55', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT55.html'},
    "GT8-Glycogenin" : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT8-1"          : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT25"           : {'CAZy Name': 'GT25', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '6  ', 'More Info': 'http://www.cazy.org/GT25.html'},
    "GT2-DPM_like"   : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT31-fringe"    : {'CAZy Name': 'GT31', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT31.html'},
    "GT2-Bact_puta"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT84"           : {'CAZy Name': 'GT84', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT84.html'},
    "GT13"           : {'CAZy Name': 'GT13', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '6  ', 'More Info': 'http://www.cazy.org/GT13.html'},
    "GT43-cele"      : {'CAZy Name': 'GT43', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT43.html'},
    "GT2-Bact_LPS1"  : {'CAZy Name': 'GT92', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT2-Bact_Oant"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT67"           : {'CAZy Name': 'GT67', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT67.html'},
    "GT2-HAS"        : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT82"           : {'CAZy Name': 'GT82', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '7  ', 'More Info': 'http://www.cazy.org/GT82.html'},
    "GT24"           : {'CAZy Name': 'GT24', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT24.html'},
    "GT31-plant"     : {'CAZy Name': 'GT31', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT31.html'},
    "GT81-Bact"      : {'CAZy Name': 'GT81', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT81.html'},
    "GT2-Bact_gt25Me": {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT2-B3GntL"     : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '4  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT49"           : {'CAZy Name': 'GT49', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT49.html'},
    "GT34"           : {'CAZy Name': 'GT34', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT34.html'},
    "GT45"           : {'CAZy Name': 'GT45', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT45.html'},
    "GT32-lower"     : {'CAZy Name': 'GT32', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT32.html'},
    "GT88"           : {'CAZy Name': 'GT88', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT88.html'},
    "GT21"           : {'CAZy Name': 'GT21', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT21.html'},
    "GT2-DPG_synt"   : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT43-b3gat2"    : {'CAZy Name': 'GT43', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT43.html'},
    "GT2-Chitin_synt": {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '5  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT8-Bact"       : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT8-Met2"       : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT2-Bact_Chlor1": {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT54"           : {'CAZy Name': 'GT54', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '6  ', 'More Info': 'http://www.cazy.org/GT54.html'},
    "GT2-Cel_bre3"   : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT2-Bact_Rham"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT6"            : {'CAZy Name': 'GT6 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT6.html' },
    "GT2-Bact_puta2" : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT7-1"          : {'CAZy Name': 'GT7 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '5  ', 'More Info': 'http://www.cazy.org/GT7.html' },
    "GT2-Csl"        : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '4  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT2-ExoU"       : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT2-Csl2"       : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '4  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT64"           : {'CAZy Name': 'GT64', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT64.html'},
    "GT2-Bact_Chlor2": {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT78"           : {'CAZy Name': 'GT78', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT78.html'},
    "GT12"           : {'CAZy Name': 'GT12', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT12.html'},
    "GT31-gnt"       : {'CAZy Name': 'GT31', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT31.html'},
    "GT2-Bact_CHS"   : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '5  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT62"           : {'CAZy Name': 'GT62', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '3  ', 'More Info': 'http://www.cazy.org/GT62.html'},
    "GT8-Met_Pla"    : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT15"           : {'CAZy Name': 'GT15', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT15.html'},
    "GT43-b3gat1"    : {'CAZy Name': 'GT43', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT43.html'},
    "GT31-b3glt"     : {'CAZy Name': 'GT31', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '8  ', 'More Info': 'http://www.cazy.org/GT31.html'},
    "GT2-CesA1"      : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '1  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT60"           : {'CAZy Name': 'GT60', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '5  ', 'More Info': 'http://www.cazy.org/GT60.html'},
    "GT14"           : {'CAZy Name': 'GT14', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '7  ', 'More Info': 'http://www.cazy.org/GT14.html'},
    "GT2-Bact_DPM_sy": {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '2  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT17"           : {'CAZy Name': 'GT17', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '7  ', 'More Info': 'http://www.cazy.org/GT17.html'},
    "GT2-Bact_LPS2"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '3  ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT77"           : {'CAZy Name': 'GT77', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT77.html'},
    "GT2-Bact_EpsO"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': '   ', 'More Info': 'http://www.cazy.org/GT2.html' },
    "GT43-b3gat3"    : {'CAZy Name': 'GT43', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT43.html'},
    "GT8-Fun"        : {'CAZy Name': 'GT8 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Retaining', 'Clade': '9  ', 'More Info': 'http://www.cazy.org/GT8.html' },
    "GT75"           : {'CAZy Name': 'GT75', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT75.html'},
    "GT2-Bact_GlfT"  : {'CAZy Name': 'GT2 ', 'Alternative Name': '', 'Fold': 'A', 'Mechanism': 'Inverting', 'Clade': 'N/A', 'More Info': 'http://www.cazy.org/GT2.html' },

}





def get_family_info(family_name):
    family_info = glycosyltransferase_db.get(family_name, {})
    
    output = ""
    for key, value in family_info.items():
        if key == "more_info":
            output += "**{}:**".format(key.title().replace("_", " ")) + "\n"
            for link in value:
                output += "[{}]({})  ".format(link, link)
        else:
            output += "**{}:** {}  ".format(key.title().replace("_", " "), value)
    
    return output


def fig_to_img(fig):
    """Converts a matplotlib figure to a PIL Image and returns it"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

def preprocess_protein_sequence(protein_fasta):
    lines = protein_fasta.split('\n')

    headers = [line for line in lines if line.startswith('>')]
    if len(headers) > 1:
        return None, "Multiple fasta sequences detected. Please upload a fasta file with only one sequence."

    protein_sequence = ''.join(line for line in lines if not line.startswith('>'))
    
    # Check for invalid characters
    valid_characters = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")  # the 20 standard amino acids
    if not set(protein_sequence).issubset(valid_characters):
        return None, "Invalid protein sequence. It contains characters that are not one of the 20 standard amino acids. Does your sequence contain gaps?"

    return protein_sequence, None


def process_family_sequence(protein_fasta):
    protein_sequence, error_msg = preprocess_protein_sequence(protein_fasta)
    if error_msg:
        return None, None, None, error_msg

    encoded_input = tokenizer([protein_sequence], padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_idsfam = encoded_input["input_ids"]
    attention_maskfam = encoded_input["attention_mask"]

    with torch.no_grad():
        outputfam = modelfam(input_idsfam, attention_mask=attention_maskfam)
        logitsfam = outputfam.logits
        probabilitiesfam = F.softmax(logitsfam, dim=1)
        _, predicted_labelsfam = torch.max(logitsfam, dim=1)

    decoded_labelsfam = yfam.inverse_transform(predicted_labelsfam.tolist())
    family_info = get_family_info(decoded_labelsfam[0])

    figfam = plt.figure(figsize=(10, 5))
    labelsfam = yfam.classes_
    probabilitiesfam = probabilitiesfam.tolist()

    # Convert the nested list to a flat list of probabilities
    probabilitiesfam_flat = probabilitiesfam[0] if probabilitiesfam else []

    # Sort labels and probabilities by probability
    labels_probsfam = list(zip(labelsfam, probabilitiesfam_flat))
    labels_probsfam.sort(key=lambda x: x[1], reverse=True)

    # Select the top 5 fams
    labels_probs_top5fam = labels_probsfam[:5]
    labels_top5, probabilities_top5 = zip(*labels_probs_top5fam)

    y_posfam = np.arange(len(labels_top5))

    plt.barh(y_posfam, [prob*100 for prob in probabilities_top5], align='center', alpha=0.5)
    plt.yticks(y_posfam, labels_top5)
    plt.xlabel('Probability (%)')
    plt.title('Top 5 Family Class Probabilities')
    plt.xlim(0, 100)
    plt.close(figfam)

    img = fig_to_img(figfam)

    if len(protein_sequence) < 100:
        return decoded_labelsfam[0], img, None, f"**Warning:** The sequence is relatively short. Fragmentary and partial sequences may result in incorrect predictions. \n\n {family_info}"


    return decoded_labelsfam[0], img, None, family_info


def process_single_sequence(protein_fasta): #, protein_file
    protein_sequence, error_msg = preprocess_protein_sequence(protein_fasta)
    if error_msg:
        return None, None, None, error_msg

    encoded_input = tokenizer([protein_sequence], padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = encoded_input["input_ids"]
    attention_mask = encoded_input["attention_mask"]

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        dprobabilities = F.softmax(logits, dim=1)[0]
        _, predicted_labels = torch.max(logits, dim=1)

    decoded_labels = label_encoder.inverse_transform(predicted_labels.tolist())
    family_info = get_family_info(decoded_labels[0])

    fig = plt.figure(figsize=(10, 5))
    labels = label_encoder.classes_
    dprobabilities = dprobabilities.tolist()

    # Sort labels and probabilities by probability
    labels_probs = list(zip(labels, dprobabilities))
    labels_probs.sort(key=lambda x: x[1], reverse=True)

    # Select the top 3 donors
    labels_probs_top3 = labels_probs[:3]
    labels_top3, probabilities_top3 = zip(*labels_probs_top3)
    
    y_pos = np.arange(len(labels_top3))
    
    plt.barh(y_pos, [prob*100 for prob in probabilities_top3], align='center', alpha=0.5)
    plt.yticks(y_pos, labels_top3)
    plt.xlabel('Probability (%)')
    plt.title('Top 3 Donor Class Probabilities')
    plt.xlim(0, 100) 
    plt.close(fig)

    img = fig_to_img(fig)

    if len(protein_sequence) < 100:
        return decoded_labels[0], img, None, f"**Warning:** The sequence is relatively short. Fragmentary and partial sequences may result in incorrect predictions. \n\n {family_info}"


    return decoded_labels[0], img, None, None

def process_sequence_file(protein_file):  # added progress parameter that is displayed in gradio #, progress=gr.Progress()
    try:
        records = list(SeqIO.parse(protein_file.name, "fasta"))
    except Exception as e:
        return str(e)

    if not os.path.exists('results'):
        os.makedirs('results')

    total = len(records)  

    for idx, record in enumerate(records):
        protein_sequence = str(record.seq)

        valid_characters = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy")  
        if not set(protein_sequence).issubset(valid_characters):
            with open(f'results/result_{idx+1}.txt', 'w') as file:
                file.write("Invalid protein sequence. It contains characters that are not one of the 20 standard amino acids. Does your sequence contain gaps?")
            continue

        label, img, _, info = process_single_sequence(protein_sequence)
        img.save(f'results/result_{idx+1}.png')
        with open(f'results/result_{idx+1}.txt', 'w') as file:
            file.write(f'Predicted Donor: {label}\n\n{info}')

        # progress(idx/total)  # Update the progress bar
    
    # Create a zip file w/ results -- To Do: Figure out how to improve compression for large files
    with zipfile.ZipFile('predicted_results.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('results/'):
            for file in files:
                zipf.write(os.path.join(root, file))

    return 'predicted_results.zip' #Provide indication of how to interpret downloaded zip file? f"**Warning:** The sequence is relatively short. Fragmentary and partial sequences may result in incorrect predictions.

    # Function to mask a residue at a particular position
def mask_residue(sequence, position):
    return sequence[:position] + 'X' + sequence[position+1:]

def generate_heatmap(protein_fasta):
    protein_sequence, error_msg = preprocess_protein_sequence(protein_fasta)

    # Tokenize and predict for original sequence
    encoded_input = tokenizer([protein_sequence], padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        original_output = model(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
    original_probabilities = F.softmax(original_output.logits, dim=1).cpu().numpy()[0]
    
    # Define the size of each group
    group_size = 10  # allow user to change this

    # Calculate the number of groups
    num_groups = len(protein_sequence) // group_size + (len(protein_sequence) % group_size > 0)

    # Initialize an array to hold the importance scores
    importance_scores = np.zeros((num_groups, len(original_probabilities)))

    # Initialize tqdm progress bar
    # with tqdm(total=num_groups, desc="Processing groups", position=0, leave=True) as pbar:
    #     # Loop through each group of residues in the sequence
    for i in range(0, len(protein_sequence), group_size):
        # Mask the residues in the group at positions [i, i + group_size)
        masked_sequence = protein_sequence[:i] + 'X' * min(group_size, len(protein_sequence) - i) + protein_sequence[i + group_size:]
        
        # Tokenize and predict for the masked sequence
        encoded_input = tokenizer([masked_sequence], padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            masked_output = model(encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
        masked_probabilities = F.softmax(masked_output.logits, dim=1).cpu().numpy()[0]
        
        # Calculate the change in probabilities and store it as the importance score
        group_index = i // group_size
        importance_scores[group_index, :] = np.abs(original_probabilities - masked_probabilities)
            
        progress = (i // group_size + 1) / num_groups * 100
        print(f"Progress: {progress:.2f}%")
    
    figmap, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(importance_scores, annot=True, cmap="coolwarm", xticklabels=label_encoder.classes_, yticklabels=[f"{i}-{i+group_size-1}" for i in range(0, len(protein_sequence), group_size)], ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("Residue Position Groups")

    img = fig_to_img(figmap)

    return img


def main_function_single(sequence, show_explanation):
    # Process seq, and return outputs for both fam and don
    family_label, family_img, _, family_info = process_family_sequence(sequence)
    donor_label, donor_img, *_ = process_single_sequence(sequence)
    figmap = None
    if show_explanation:
        figmap = generate_heatmap(sequence)
    return family_label, family_img, family_info, donor_label, donor_img, figmap

def main_function_upload(protein_file): #, progress=gr.Progress()
    return process_sequence_file(protein_file) #, progress

prediction_imagefam = gr.outputs.Image(type='pil', label="Family prediction graph")
prediction_imagedonor = gr.outputs.Image(type='pil', label="Donor prediction graph")
prediction_explain = gr.outputs.Image(type='pil', label="Donor prediction explanation")


with gr.Blocks() as app:
    gr.Markdown("# Glydentify (alpha v0.3)")

    with gr.Tab("Single Sequence Prediction"):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                sequence = gr.inputs.Textbox(lines=16, placeholder='Enter Protein Sequence Here...', label="Protein Sequence")
                explanation_checkbox = gr.inputs.Checkbox(label="Show Explanation", default=False)
            with gr.Column():
                with gr.Accordion("Example:"):
                    gr.Markdown("""
                                \>sp|Q9Y5Z6|B3GT1_HUMAN Beta-1,3-galactosyltransferase 1 OS=Homo sapiens OX=9606 GN=B3GALT1 PE=1 SV=1  
                                MASKVSCLYVLTVVCWASALWYLSITRPTSSYTGSKPFSHLTVARKNFTFGNIRTRPINPHSFEFLINEPNKCEKNIPFLVILIST  
                                THKEFDARQAIRETWGDENNFKGIKIATLFLLGKNADPVLNQMVEQESQIFHDIIVEDFIDSYHNLTLKTLMGMRWVATFCSK  
                                AKYVMKTDSDIFVNMDNLIYKLLKPSTKPRRRYFTGYVINGGPIRDVRSKWYMPRDLYPDSNYPPFCSGTGYIFSADVAELIYK  
                                TSLHTRLLHLEDVYVGLCLRKLGIHPFQNSGFNHWKMAYSLCRYRRVITVHQISPEEMHRIWNDMSSKKHLRC  
                                """)
                family_prediction = gr.outputs.Textbox(label="Predicted family")
                donor_prediction = gr.outputs.Textbox(label="Predicted donor")
                info_markdown = gr.Markdown()

        # Predict and Clear buttons
        with gr.Row().style(equal_height=True):
            with gr.Column():
                predict_button = gr.Button("Predict")
                predict_button.click(main_function_single, inputs=[sequence, explanation_checkbox],
                                     outputs=[family_prediction, prediction_imagefam, info_markdown,
                                              donor_prediction, prediction_imagedonor, prediction_explain])

        # Family & Donor Section
        with gr.Row().style(equal_height=True):
            with gr.Column():
                with gr.Accordion("Prediction Bar Graphs:"):
                    prediction_imagefam.render() # = gr.outputs.Image(type='pil', label="Family prediction graph")
                    prediction_imagedonor.render() # = gr.outputs.Image(type='pil', label="Donor prediction graph")

            # Explain Section
            with gr.Column():
                if explanation_checkbox:  # Only render if the checkbox is checked
                    with gr.Accordion("Donor explanation"):
                        prediction_explain.render() # = gr.outputs.Image(type='pil', label="Donor prediction explaination")
    
    with gr.Tab("Multiple Sequence Prediction"):
        with gr.Row().style(equal_height=True):
            with gr.Column():
                protein_file = gr.inputs.File(label="Upload FASTA file")
            with gr.Column():
                result_file = gr.outputs.File(label="Download predictions of uploaded sequences")
                with gr.Row().style(equal_height=True):
                    with gr.Column():
                        process_button = gr.Button("Process")
                        process_button.click(main_function_upload, inputs=protein_file, outputs=[result_file])                        
                    with gr.Column():
                        clear = gr.Button("Clear")
                        clear.click(lambda: None) 
            # clear.click()

app.launch(show_error=True)



