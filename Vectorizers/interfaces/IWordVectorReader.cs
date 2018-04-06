using System;

namespace BotBotNLP.Vectorizers {
  interface IWordVectorReader {
    UInt64 MaxWords { get; }
    int EmbeddingDim { get; }
    double[] GetWordVector(string word);
  }
}