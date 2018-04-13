using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Linq;
using System.IO;
using BotBotNLP.DataLoader.Models;
using BotBotNLP.NeuralNetwork.Sparse;
using BotBotNLP.Vectorizers;
using Newtonsoft.Json;

namespace BotBotNLP.DataLoader
{
  public class IntentsLoader
  {
    public string FileName { get; private set; }

    public List<Intent> Intents { get; private set; }

    public SentenceVectorizer Vectorizer {get; set;}

    public int Count {
      get {
        return this.Intents
          .Select(intent => intent.usersays.Count)
          .Sum();
      }
    }

    public int InputDimensions {
      get {
        return this.Vectorizer.SentenceEmbeddingDim;
      }
    }

    public int OutputDimensions {
      get {
        return this.Intents.Count;
      }
    }

    public IntentsLoader(string fileName, SentenceVectorizer vectorizer)
    {
      this.FileName = fileName;
      this.Vectorizer = vectorizer;
      this.Initialize(FileName);
    }

    private void Initialize(string FileName)
    {
      var serializer = new JsonSerializer();
      using (var sr = new StreamReader(FileName))
      {
        using (var reader = new JsonTextReader(sr)) {
          this.Intents = serializer.Deserialize<List<Intent>>(reader);
        }
      }
    }

    public static SparseVector<double> ToCategorical(int value, int length) {
      var result = new SparseVector<double>(length);
      result[value] = 1d;
      return result;
    }

    public Tuple<SparseMatrix<double>, SparseMatrix<double>> GetData() {
      var examplesCount = this.Count;
      var classes = this.OutputDimensions;
      
      var inputs = new SparseMatrix<double>(examplesCount, this.Vectorizer.SentenceEmbeddingDim);
      var targets = new SparseMatrix<double>(examplesCount, classes);
      var count = 0;
      for (var i = 0; i < classes; i++) {
        foreach(var sentence in this.Intents[i].usersays) {
          inputs[count] = this.Vectorizer.SentenceToVector(sentence);
          targets[count] = ToCategorical(i, classes);
          count++;
        }
      }

      return new Tuple<SparseMatrix<double>, SparseMatrix<double>>(inputs, targets);
    }
  }
}