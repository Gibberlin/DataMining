import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_PATH = ROOT_DIR / "Datasets" / "Assignment-2" / "dataset.csv"

data={}
total_not_spam=0
total_spam=0

if not DATASET_PATH.exists():
    raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")

with open(DATASET_PATH, 'r') as file:
    reader=csv.DictReader(file)
    for row in reader:
        word=row["word"].strip().lower()
        ns=int(row["not_spam"])
        sp=int(row["spam"])

        data[word]={"not_spam":ns,"spam":sp}
        total_not_spam+=ns
        total_spam+=sp

total_words=total_not_spam+total_spam
p_not_spam=total_not_spam/total_words
p_spam=total_spam/total_words

vocab_size=len(data)

def classify(text):
    words=text.lower().split()

    prob_not_spam=p_not_spam
    prob_spam=p_spam

    for word in words:
        if word in data:
            prob_not_spam*=(data[word]["not_spam"]+1)/(total_not_spam+vocab_size)
            prob_spam*=(data[word]["spam"]+1)/(total_spam+vocab_size)
        else:
            prob_not_spam*=1/(total_not_spam+vocab_size)
            prob_spam*=1/(total_spam+vocab_size)
    return prob_not_spam, prob_spam
    

if __name__=="__main__":
    print("Enter a message to classify as 'spam' or 'not spam':")
    message=input("Enter message: ")
    ns, sp=classify(message)
    print(f"Probability(Not Spam)...: {ns:.10f}")
    print(f"Probability(Spam)....:{sp:.10f}")
    if ns>sp:
        print("\nThe message is classified as: NOT SPAM")
    elif sp>ns:
        print("\nThe message is classified as: SPAM")
    else:
        print("\nThe message is classified as: UNKNOWN")
