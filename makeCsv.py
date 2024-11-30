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

    # Separate the classes
    df_minority = df[df['label'] == 1]
    df_majority = df[df['label'] == 0]

    # Determine the number of samples to match the majority class
    n_samples = df_majority.shape[0]

    # Oversample the minority class
    df_minority_oversampled = df_minority.sample(n=n_samples, replace=True, random_state=42)

    # Combine back to a single DataFrame
    df_balanced = pd.concat([df_majority, df_minority_oversampled])

    # Shuffle the DataFrame
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    # Print the number of samples for each class
    class_counts = df_balanced['label'].value_counts()

    print("Class distribution after balancing:")
    print(class_counts)

    # Optionally print the counts more explicitly
    print(f"Class 0: {class_counts[0]} samples")
    print(f"Class 1: {class_counts[1]} samples")
    
    df_balanced.to_csv(output_csv, index=False)
    print(f"Combined dataset saved to {output_csv}")

if __name__ == '__main__':
    create_dataset()
