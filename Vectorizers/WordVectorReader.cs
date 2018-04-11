using System;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.CompilerServices;

namespace BotBotNLP.Vectorizers
{
  class WordVectorReader : IWordVectorReader
  {
    public string FileName { get; private set; }
    public UInt64 MaxWords { get; private set; }

    public int EmbeddingDim { get; private set; }
    private string[] wordsDictionary;
    private double[][] embeddingVector;

    public WordVectorReader(string fileName, UInt64 maxWords = 200000)
    {
      this.FileName = fileName;
      this.MaxWords = maxWords;
      if (!this.Initialize()) {
        Console.WriteLine("Warning: Error when initializing word vectors");
      };
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public double[] GetWordVector(string word)
    {
      var idx = Array.BinarySearch(this.wordsDictionary, word.Trim().ToLowerInvariant());
      if (idx < 0)
      {
        return new double[EmbeddingDim];
      }
      else
      {
        return this.embeddingVector[idx];
      }
    }

    private bool Initialize()
    {
      if (!File.Exists(this.FileName))
      {
        return false;
      }
      else
      {
        try
        {
          UInt64 idx = 0;

          using (var reader = new StreamReader(this.FileName))
          {
            while (!reader.EndOfStream)
            {
              var tmp = reader.ReadLine().Split(' ');
              if (idx == 0)
              {
                this.EmbeddingDim = tmp.Length - 1;
                this.embeddingVector = new double[this.MaxWords][];
                this.wordsDictionary = new string[this.MaxWords];
              }

              this.wordsDictionary[idx] = tmp[0].ToLowerInvariant();
              // LINQ version
              // this.embeddingVector[idx] = tmp
              //     .Skip(1)
              //     .Select(element => double.Parse(element, CultureInfo.InvariantCulture.NumberFormat))
              //     .ToArray();
              this.embeddingVector[idx] = new double[this.EmbeddingDim];
              for (var i = 1; i < tmp.Length; i++) {
                this.embeddingVector[idx][i - 1] = double.Parse(tmp[i], CultureInfo.InvariantCulture.NumberFormat);
              }
              
              idx += 1;
              if (idx >= this.MaxWords)
              {
                break;
              }
            }
          }

          // Sort for binary search for faster retrieval
          Array.Sort(this.wordsDictionary, this.embeddingVector);

          return true;
        }
        catch
        {
          return false;
        }
      }
    }
  }
}
