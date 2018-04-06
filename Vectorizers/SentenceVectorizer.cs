using System;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using System.Security.Cryptography;

namespace BotBotNLP.Vectorizers
{
  class SentenceVectorizer {
    public UInt64 HashingBins {get; private set;}
    public IWordVectorReader WordVectorReader {get; set;}
    
    public SentenceVectorizer(IWordVectorReader wordVectorReader, UInt64 hashingBins = 10000000) {
      this.WordVectorReader = wordVectorReader;
      this.HashingBins = hashingBins;
    }

    private Regex word_tokenize = new Regex(@"[a-zA-Z]+|\d+|[^a-zA-Z\d\s]+",
      RegexOptions.Compiled | RegexOptions.CultureInvariant);
    public string[] SentenceToWords(string sentence) {
      return this.word_tokenize
        .Matches(sentence)
        .Select(match => match.Value)
        .ToArray();
    }

    private UInt64 CalculateHash(string input) {
      UInt64 hashedValue = 3074457345618258791ul;
      for(int i=0; i < input.Length; i++)
      {
          hashedValue += input[i];
          hashedValue *= 3074457345618258799ul;
      }
      return hashedValue;
    }

    private string[] GetBiGramList(string[] words) {
      var bigrams = new List<string>();
      if (words.Length > 2) {
        for (var i = 1; i < words.Length; i++) {
          bigrams.Add(words[i-1] + words[i]);
        }
      }
      return bigrams.ToArray();
    }

    public double[] SentenceToVector(string sentence, bool useHashingTrick = true) {
      var words = this.SentenceToWords(sentence.Trim().ToLowerInvariant());
      if (words.Length == 0) {
        return null;
      }
      else {
        var wordvec_embeds = new double[this.WordVectorReader.EmbeddingDim];
        // Sum all word vectors
        foreach (var word in words) {
          var wordEmbeds = this.WordVectorReader.GetWordVector(word);
          for (var i = 0; i < this.WordVectorReader.EmbeddingDim; i++) {
            wordvec_embeds[i] += wordEmbeds[i];
          }
        }

        if (words.Length > 1) {
          for (var i = 0; i < this.WordVectorReader.EmbeddingDim; i++) {
            wordvec_embeds[i] = wordvec_embeds[i] / words.Length;
          }
        }
        
        if (!useHashingTrick) {
          return wordvec_embeds;
        }
        else {
          var wordvec_dim = (UInt64)this.WordVectorReader.EmbeddingDim;
          var embeddingDim =  wordvec_dim + this.HashingBins;
          
          var embeds = new double[embeddingDim];
          Array.Copy(wordvec_embeds, embeds, this.WordVectorReader.EmbeddingDim);

          if (words.Length > 2) {
            var bigrams = this.GetBiGramList(words);
            Parallel.ForEach(bigrams, bigram => {
              var hash = this.CalculateHash(bigram);
              var hash_loc = hash % (this.HashingBins - 1) + 1;
              embeds[wordvec_dim + hash_loc] = 1;
            });
          }

          return embeds;
        }
      }
    }
  }
}