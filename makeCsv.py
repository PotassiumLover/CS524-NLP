import os
import pandas as pd

def create_dataset(books_dir='books', dataset_dir='dataset', output_csv='combined_dataset.csv'):
    texts = []
    labels = []
    
    def read_text_file(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None

    # Process 'books' directory (label 1)
    for filename in os.listdir(books_dir):
        filepath = os.path.join(books_dir, filename)
        if filename.endswith('.txt') and os.path.isfile(filepath):
            text = read_text_file(filepath)
            if text:
                texts.append(text)
                labels.append(1)
    
    # Process 'dataset' directory (label 0)
    for filename in os.listdir(dataset_dir):
        filepath = os.path.join(dataset_dir, filename)
        if filename.endswith('.txt') and os.path.isfile(filepath):
            text = read_text_file(filepath)
            if text:
                texts.append(text)
                labels.append(0)
    
    df = pd.DataFrame({'text': texts, 'label': labels})
    
    # Shuffle the DataFrame to mix the classes
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Combined dataset saved to {output_csv}")

if __name__ == '__main__':
    create_dataset()
