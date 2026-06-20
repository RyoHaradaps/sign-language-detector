import pandas as pd
import re
from collections import Counter
import os

# Load the iSign metadata
print("Loading iSign metadata...")
df = pd.read_csv('data/isign/iSign_v1.1.csv')
print(f"Total sentences: {len(df):,}")

# Define ISL-related words to extract
isl_words = {
    'namaste': ['namaste', 'namaskar', 'greeting', 'hello', 'hi'],
    'thank_you': ['thank', 'thanks', 'thank you', 'grateful'],
    'please': ['please', 'pls', 'kindly'],
    'sorry': ['sorry', 'apologize', 'apology', 'regret'],
    'help': ['help', 'assist', 'aid', 'support'],
    'yes': ['yes', 'yeah', 'yep', 'sure', 'agree', 'confirm'],
    'no': ['no', 'not', 'never', 'none', 'deny', 'refuse'],
    'stop': ['stop', 'halt', 'cease', 'end', 'finish'],
    'water': ['water', 'drink', 'thirsty'],
    'food': ['food', 'eat', 'meal', 'hungry', 'dinner', 'lunch', 'breakfast'],
    'good': ['good', 'great', 'awesome', 'excellent', 'nice'],
    'bad': ['bad', 'terrible', 'awful', 'poor'],
    'love': ['love', 'like', 'adore', 'cherish'],
    'friend': ['friend', 'buddy', 'pal', 'companion'],
    'family': ['family', 'relative', 'home'],
    'work': ['work', 'job', 'career', 'profession'],
    'school': ['school', 'college', 'university', 'study', 'learn'],
    'time': ['time', 'hour', 'minute', 'second', 'clock'],
    'today': ['today', 'now', 'current', 'present'],
    'tomorrow': ['tomorrow', 'future', 'next day'],
    'yesterday': ['yesterday', 'past', 'previous day']
}

def extract_matches(text):
    """Check if a sentence contains any ISL words"""
    if not isinstance(text, str):
        return []
    text_lower = text.lower()
    matches = []
    for word, keywords in isl_words.items():
        for kw in keywords:
            if kw in text_lower:
                matches.append(word)
                break
    return matches

# Apply extraction
print("Extracting ISL words from sentences...")
df['matched_words'] = df['text'].apply(extract_matches)

# Filter rows that have matches
matched_df = df[df['matched_words'].apply(len) > 0]
print(f"Sentence count with ISL words: {len(matched_df):,}")

# Show some examples
print("\n📝 Examples of matched sentences:")
for _, row in matched_df.head(30).iterrows():
    matches = ', '.join(row['matched_words'])
    preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
    print(f"  {preview} → {matches}")

# Count occurrences of each word
all_matches = [word for sublist in matched_df['matched_words'] for word in sublist]
word_counts = Counter(all_matches)

print("\n📊 ISL Word Frequency in iSign:")
print("-" * 50)
for word, count in word_counts.most_common():
    print(f"  {word:12} → {count:>8,} occurrences")

# Save word frequency report
with open('data/isign/isign_word_frequency.txt', 'w') as f:
    f.write("ISL Word Frequency in iSign Dataset\n")
    f.write("=" * 50 + "\n\n")
    for word, count in word_counts.most_common():
        f.write(f"{word:15} {count:>8,}\n")

print(f"\n✅ Word frequency saved to: data/isign/isign_word_frequency.txt")

# Save filtered dataset
matched_df.to_csv('data/isign/isign_filtered_dataset.csv', index=False)
print(f"✅ Filtered dataset saved to: data/isign/isign_filtered_dataset.csv")

# Summary
print("\n📊 Summary:")
print(f"  Total sentences in iSign: {len(df):,}")
print(f"  Sentences with ISL words: {len(matched_df):,}")
print(f"  Unique ISL words found: {len(word_counts)}")
print(f"  Total word occurrences: {sum(word_counts.values()):,}")