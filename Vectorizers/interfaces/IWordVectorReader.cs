using System;

namespace BotBotNLP.Vectorizers {
  public interface IWordVectorReader {
    UInt64 MaxWords { get; }
    int EmbeddingDim { get; }
    double[] GetWordVector(string word);
  }
}