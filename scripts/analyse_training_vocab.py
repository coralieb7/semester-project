"""
FILE: scripts/analyze_training_vocab.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import Counter
from config.paths import Paths

# Load pairing file
pairing_path = Paths.TRAINING_DATA_DIR / "pairing.json"
with open(pairing_path, 'r') as f:
    pairings = json.load(f)

# Extract all captions
captions = [data['text_prompt'] for data in pairings.values()]

print(f"Analyzing {len(captions)} training captions...\n")

# Count word frequency
all_words = []
for caption in captions:
    # Simple word extraction (lowercase, split)
    words = caption.lower().replace(',', '').replace('.', '').split()
    all_words.extend(words)

word_freq = Counter(all_words)

# Find common style keywords
print("="*70)
print("TOP 30 MOST COMMON WORDS IN TRAINING DATA")
print("="*70)
for word, count in word_freq.most_common(30):
    percentage = (count / len(captions)) * 100
    print(f"{word:20s} : {count:3d} times ({percentage:.1f}% of captions)")

# Find bigrams (two-word phrases)
bigrams = []
for caption in captions:
    words = caption.lower().split()
    for i in range(len(words) - 1):
        bigrams.append(f"{words[i]} {words[i+1]}")

bigram_freq = Counter(bigrams)

print("\n" + "="*70)
print("TOP 20 MOST COMMON PHRASES")
print("="*70)
for phrase, count in bigram_freq.most_common(20):
    percentage = (count / len(captions)) * 100
    print(f"{phrase:30s} : {count:3d} times ({percentage:.1f}% of captions)")

# Suggest trigger phrase
print("\n" + "="*70)
print("SUGGESTED TRIGGER PHRASES")
print("="*70)

# Find words that appear in >50% of captions
core_words = [word for word, count in word_freq.items() 
              if count > len(captions) * 0.5 and len(word) > 3]

print("\nCore style words (appear in >50% of captions):")
print(", ".join(sorted(core_words)))

print("\n✨ RECOMMENDED TRIGGER PHRASE:")
# Common style descriptors
style_keywords = []
for word in ['minimalist', 'graphic', 'design', 'centered', 'modern', 
             'flat', 'vector', 'clean', 'simple', 'abstract', 'white background']:
    if word in word_freq and word_freq[word] > len(captions) * 0.3:
        style_keywords.append(word)

if style_keywords:
    trigger = " ".join(style_keywords)
    print(f'  "{trigger}"')
    print(f"\nUse this in all your generation prompts to activate the LoRA!")
else:
    print("  (Analyze the lists above to create your own)")