import re
import wikipedia
import math

from collections import Counter
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize

class NGramModel:
  def __init__(self, text):
    self.text = text
    self.tokens = self._preprocess_text(text)
    self.bigram_counts = Counter(bigrams(self.tokens))
    self.trigram_counts = Counter(trigrams(self.tokens))
    self.unigram_counts = Counter(self.tokens)

  def _preprocess_text(self, text):
    """Cleans and tokenizes text."""
    text = text.lower() 
    text = re.sub(r'[^a-z\s]', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return word_tokenize(text)
  
  def __bigram_probabilities(self):
    bigram_probs = { bigram: round(count / self.unigram_counts[bigram[0]], 3) for bigram, count in self.bigram_counts.items() }
    return bigram_probs
  
  def __trigram_probabilities(self):
    trigram_probs = { trigram: round(count / self.bigram_counts[trigram[0], trigram[1]], 3) for trigram, count in self.trigram_counts.items() }
    return trigram_probs

  def generate_text(self, start_word, len = 10, n_gram = 'bigram'):
    current_word = start_word.lower()
    generated_text = [current_word]

    for _ in range(len):
      if (n_gram == 'bigram'):
        candidates = { k[1]: v for k, v in self.__bigram_probabilities().items() if k[0] == current_word }
      elif (n_gram == 'trigram'):
        candidates = { k[2]: v for k, v in self.__trigram_probabilities().items() if k[0] == current_word or k[1] == current_word }

      next_word = max(candidates, key=candidates.get)
      generated_text.append(next_word)
      current_word = next_word
    
    return ' '.join(generated_text)
  
  def perplexity(self, gen_text, n_gram='bigram', smoothing=1e-6):
    """Calculates perplexity of generated text using bigram or trigram model."""
    
    gen_tokens = word_tokenize(gen_text, language='english')
    
    if n_gram == 'bigram':
        all_bigram_probs = self.__bigram_probabilities()
        gen_token_probs = [
            all_bigram_probs.get((gen_tokens[i-1], gen_tokens[i]), smoothing)
            for i in range(1, len(gen_tokens))
        ]
    elif n_gram == 'trigram':
        all_trigram_probs = self.__trigram_probabilities()
        gen_token_probs = [
            all_trigram_probs.get((gen_tokens[i-2], gen_tokens[i-1], gen_tokens[i]), smoothing)
            for i in range(2, len(gen_tokens))
        ]
    else:
        raise ValueError("Invalid n-gram type. Use 'bigram' or 'trigram'.")

    print(f'N-Gram Probabilities: {gen_token_probs}')
    
    log_sum = sum(math.log(prob) for prob in gen_token_probs)
    
    N = len(gen_token_probs)
    
    # Perplexity formula
    return round(math.exp(-log_sum / N), 4) if N > 0 else float('inf')


def main():
  page = wikipedia.page('Banach-Tarski paradox')
  wiki_text = page.content[:8004].lower() # This is approximately 1159 words, including new lines.
  wiki_text = re.sub(r'[=]', ' ', wiki_text) # Remove headings

  ngram_model = NGramModel(wiki_text)

  # Use the below code to print the probabilities and counts of bigrams and trigrams
  # for trigram, prob in ngram_model.trigram_probabilities().items():
  #   print(f"P({trigram[2]} | {trigram[0]}, {trigram[1]}) = {prob: .4f}")
  #   print(f"C({trigram[0] + ', ' + trigram[1] + ', ' + trigram[2]}) = {ngram_model.trigram_counts[trigram]}")
  #   print(f"C({trigram[0] + ', ' + trigram[1]}) = {ngram_model.bigram_counts[trigram[0], trigram[1]]}\n")

  gen_text = ngram_model.generate_text('the', 8, 'bigram')
  print(f'Bi-gram Generated Text: "{gen_text}"')
  print(f'Tri-gram Perplexity: {ngram_model.perplexity(gen_text, "trigram")}\n')
  print(f'Bi-gram Perplexity: {ngram_model.perplexity(gen_text)}')


if __name__ == '__main__':
  main()