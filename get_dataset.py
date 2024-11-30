import re
import unidecode
import requests
from Book_module.Book import Book

print("test")
exit(0)
text = ""

with open("test.txt", "r") as f:
    text = f.read()

books = {
    
    "The murder of Roger Ackroyd": ["69087", "novel", r"(CHAPTER\s+[IVXLCDM]+\n\n[^\n]+?\n\n)", "Agatha Christie"],
    "The Mysterious Affair at Styles":["863", "novel", r"(CHAPTER\s+[IVXLCDM]+.\n[^\n]+?\n\n)", "Agatha Christie"],
    "POIROT INVESTIGATES" :["61262", "short_stories", r"\b[IVXLCDM]+\b\n\n\s*(.+?)\s*\n\n", "Agatha Christie", r"\w+\n+(.*?)\n+"],
    "The Murder on the Links" : ["58866", "novel", r"\n\n([0-9]+ .+?)\n\n", "Agatha Christie"],
    "THE SECRET ADVERSARY": ["1155", "novel", r"\n\n(CHAPTER [IVXLCDM]+.+?)\n\n", "Agatha Christie"],
    "The Big Four" : ["70114", "novel", r"\n\s*([0-9]+\.\s+.+?)\n", "Agatha Christie"],
    "The Mystery of the Blue Train": ["72824", "novel", r"\n\s*([0-9]+\.\s+.+?)\n", "Agatha Christie"],
    "The Man in the Brown Suit" : ["61168", "novel", r"\n\n\s+(CHAPTER [IVXLCDM]+)\n\n", "Agatha Christie"],
    "The Secret of Chimneys" : ["65238", "novel", r"\n\n([0-9]+\s+.+?)\n\n", "Agatha Christie"],
    
    "The Adventures of Sherlock Holmes" : ["1661", "short_stories", r"\n\n([IVXLCDM]+\.\s+.+?)\n\n", "Arthur Conan Doyle", r"\.(.*?)\n"],
    "A Study in Scarlet" : ["244", "novel", r"\n\n(CHAPTER\s+[IVXLCDM]+\.\n.+?)\n\n", "Arthur Conan Doyle"],
    "The Hound of the Baskervilles": ["2852", "novel", r"\n\n(Chapter\s+[0-9]+\.\n.+?)\n\n", "Arthur Conan Doyle"],
    "The Sign of the Four": ["2097", "novel", r"\n\n(Chapter\s+[IVXLCDM]+\n.+?)\n\n", "Arthur Conan Doyle"],
    "The Valley of Fear": ["3289","novel",r"\n\n(Chapter\s+[IVXLCDM]+\n.+?)\n\n","Arthur Conan Doyle"],
    "The Memoirs of Sherlock Holmes": ["834","short_stories",r"\n\n([IVXLCDM]+\..+?)\n\n","Arthur Conan Doyle", r"\.(.*?)\n"],
    "The Return of Sherlock Holmes": ["108", "short_stories", r"\n\n(THE ADVENTURE OF .+?)\n\n","Arthur Conan Doyle", r"(THE ADVENTURE OF .+?)\n"],
    "The Lost World" : ["139", "novel", r"(CHAPTER [IVXLCDM]+\n\n\s*.+?\n)", "Arthur Conan Doyle"],
    "His Last Bow: An Epilogue of Sherlock Holmes": ["2350", "short_stories", r"\n\n(The Adventure of Wisteria Lodge|The Adventure of the Bruce-Partington Plans|The Adventure of the Devil’s Foot|The Adventure of the Red Circle|The Disappearance of Lady Frances Carfax|The Adventure of the Dying Detective|His Last Bow: The War Service of Sherlock Holmes)\n\n","Arthur Conan Doyle", r"\n\n(.*?)\n\n"],
    "The White Company" : ["903", "novel",r"\n\n(CHAPTER [IVXLCDM]+\. .+?)\n\n", "Arthur Conan Doyle"],
    
    "The Extraordinary Adventures of Arsène Lupin, Gentleman-Burglar": ["6133", "short_stories", r"\n\n([IVXLCDM]+\..+?)\n\n", "Maurice Leblanc", r"\.(.*?)\n"],
    "Arsène Lupin versus Herlock Sholmes": ["40203", "novel", r"\n\n(CHAPTER [IVXLCDM]+\.\n.+?)\n\n", "Maurice Leblanc"],
    "The Hollow Needle; Further Adventures of Arsène Lupin": ["4017", "novel", r"\n\n(CHAPTER.+\n.+?)\n\n", "Maurice Leblanc"],
    "The Confessions of Arsène Lupin": ["28093", "short_stories", r"\n\n([IVXLCDM]+\n\n.+?)\n\n", "Maurice Leblanc", r"[IVXLCDM]+\n\n(.+?)\n"],
    "The Crystal Stopper": ["1563", "novel", r"\n\n(CHAPTER [IVXLCDM]+\.\n\n.+?)\n\n", "Maurice Leblanc"],
    "The Teeth of the Tiger": ["13058", "novel", r"\n\n(CHAPTER .+\n\n.+?)\n\n", "Maurice Leblanc"],
    "The Eight Strokes of the Clock": ["7896", "short_stories", r"\n\n([IVXLCDM]+\n\n.+?)\n\n", "Maurice Leblanc", r"[IVXLCDM]+\n\n(.+?)\n"],
    "813": ["4018", "novel", r"\n\n(CHAPTER [IVXLCDM]+\n\n.+?)\n\n", "Maurice Leblanc"],
    "The Golden Triangle The Return of Arsène Lupin": ["34795", "novel", r"\n\n(CHAPTER [IVXLCDM]+\n\n.+?)\n\n", "Maurice Leblanc"]
}

text = unidecode.unidecode(text)
text = text.replace("\r","")

pattern = r"\n\n(CHAPTER [IVXLCDM]+\n\n.+?)\n\n"
# test = re.findall(pattern, text)

# print(test)
# exit(0)
"""
    Project Gutenberg offers several works by Agatha Christie. Here are some of her books available on the platform, along with their respective eBook IDs:

1. **"The Murder of Roger Ackroyd"** (eBook ID: 69087)
2. **"The Mysterious Affair at Styles"** (eBook ID: 863)
3. **"Poirot Investigates"** (eBook ID: 61262)
4. **"The Murder on the Links"** (eBook ID: 58866)
5. **"The Secret Adversary"** (eBook ID: 1155)
6. **"The Man in the Brown Suit"** (eBook ID: 70115)
7. **"The Big Four"** (eBook ID: 70114)
8. **"The Secret of Chimneys"** (eBook ID: 70116)
9. **"The Mystery of the Blue Train"** (eBook ID: 72824)
"""

"""
"The Adventures of Sherlock Holmes" (eBook ID: 1661) //Done
"A Study in Scarlet" (eBook ID: 244)
"The Hound of the Baskervilles" (eBook ID: 2852)
"The Sign of the Four" (eBook ID: 2097)
"The Valley of Fear" (eBook ID: 3289)
"The Memoirs of Sherlock Holmes" (eBook ID: 834)
"The Return of Sherlock Holmes" (eBook ID: 108)
"The Lost World" (eBook ID: 139)
"His Last Bow An Epilogue of Sherlock Holmes" (eBook ID: 2350)
"The White Company" (eBook ID: 903)
     
"""

import re
import os

def process_text(book_name: str, book_meta: list[str], text: str):
    # Normalize line endings
    text = text.replace("\r", "")
    # print(text[0 : 60])
    # Remove Project Gutenberg header
    pattern_header = r"\*\*\* START OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK .*?\*\*\*"
    search_result = re.search(pattern_header, text, flags=re.DOTALL | re.IGNORECASE)
    if search_result:
        text = text[search_result.end():]
    else:
        print("Warning: Start of Project Gutenberg header not found.")
    
    # Remove Project Gutenberg footer
    pattern_footer = r"\*\*\* END OF (?:THIS |THE )?PROJECT GUTENBERG EBOOK .*?\*\*\*"
    search_result = re.search(pattern_footer, text, flags=re.DOTALL | re.IGNORECASE)
    if search_result:
        text = text[:search_result.start()]
    else:
        print("Warning: End of Project Gutenberg footer not found.")
    
    
    # Find and process all chapters
    chapters = list(re.finditer(book_meta[2], text))  # Convert to list for flexibility

    if book_meta[1] == "novel":
    
        # Handle the first chapter specially
        if chapters:
            first_chapter = chapters[0]
            # Remove everything from the start of the text to the end of the first chapter
            first_chapter_end = first_chapter.end()
        
            # Strip from start to the end of the first chapter
            text = text[first_chapter_end:]

        # Remove remaining chapter headers
        for chapter in chapters[1:]:
            text = text.replace(chapter.group(0), "")
        
        # Prepare the save file name
        save_file = book_name.lower().replace(" ", "_") + f"_[{book_meta[3].replace(" ", "+")}].txt"
        save_file = os.path.join("dataset", save_file)
        
        # Save the processed text
        # os.makedirs(os.path.dirname(save_file), exist_ok=True)  # Create the directory if it doesn't exist
        with open(save_file, "w") as f:
            f.write(text)
    elif meta[1] == "short_stories":
   
        if chapters:
            for i, chapter in enumerate(chapters):
                # Determine the start and end of the current chapter
                chapter_start = chapter.end()
                if i + 1 < len(chapters):
                    chapter_end = chapters[i + 1].start()
                else:
                    chapter_end = len(text)  # Last chapter goes to the end of the text

                # Extract the chapter text, including its title
                chapter_text = text[chapter_start:chapter_end]

                # Clean up the chapter title for the filename
                # print(chapter.group(0))
                # print(chapter)
                chapter_title = re.findall(book_meta[4], str(chapter.group(0)))[0]
                # print(chapter_title)
                # continue
                chapter_title = chapter_title.lower().strip(" ").replace(" ", "_") 
                # print(chapter_title)
                save_title = chapter_title + f"_[{book_meta[3].replace(" ", "+")}].txt"

                
                # Prepare the save file name
                save_file = os.path.join("dataset", save_title)

                print(save_file)
                # Save the chapter to its file
                with open(save_file, "w") as f:
                    f.write(chapter_text)


def get_book(url:str):
    
    response = requests.get(full_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content of the webpage
        content = response.text
    else:
        print(f"ERROR: Unable to get {url}")
        return ""
    
    return response.text

url = "https://www.gutenberg.org/cache/epub/"

for file, meta in books.items():
    
    # if file != "The Eight Strokes of the Clock":
    #     continue
    
    full_url = url + str(meta[0]) + "/pg" + str(meta[0]) + ".txt"
    text = get_book(full_url)
    if text == "":
        continue
    
    process_text(file, meta, text)

# book_list = []  
# for file in os.listdir("dataset"):
    
#     file_path = os.path.join("dataset", file)
#     book_list.append(Book(file_path))
#     book_list[-1].pre_process()
#     print(file_path)

