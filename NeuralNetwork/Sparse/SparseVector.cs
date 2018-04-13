using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Linq;

namespace BotBotNLP.NeuralNetwork.Sparse
{
  public class SparseVector<T>
  {
    public int Length { get; private set; }

    public List<int> Keys { get; private set; } = new List<int>();
    public List<T> Values { get; private set; } = new List<T>();

    public SparseVector(int length)
    {
      this.Length = length;
    }

    public T this[int index]
    {
      get
      {
        return this.Get(index);
      }
      set
      {
        this.Set(index, value);
      }
    }

    private void Set(int index, T value)
    {
      if (Keys.Count == 0)
      {
        Keys.Add(index);
        Values.Add(value);
      }
      else
      {
        var itemIndex = Keys.BinarySearch(index);
        if (itemIndex >= 0)
        {
          Values[itemIndex] = value;
        }
        else
        {
          Keys.Insert(~itemIndex, index);
          Values.Insert(~itemIndex, value);
        }
        if (Keys.Count != Values.Count) {
          Console.WriteLine("Break!");
        }
      }
    }

    private T Get(int index)
    {
      var itemIndex = Keys.BinarySearch(index);
      if (itemIndex >= 0)
      {
        return Values[itemIndex];
      }
      else
      {
        return default(T);
      }
    }

    public T[] ToDense()
    {
      var result = new T[this.Length];
      if (this.Keys.Count > 0)
      {
        for (var i = 0; i < this.Keys.Count; i++)
        {
          result[this.Keys[i]] = this.Values[i];
        }
      }
      return result;
    }

    public SparseMatrix<T> ToMatrix()
    {
        var result = new SparseMatrix<T>(1, this.Length);
        result[0] = this;
        return result;
    }

    public static void Copy(T[] sourceArray, SparseVector<T> destVector, int length = -1)
    {
      Copy(sourceArray, 0, destVector, 0, length);
    }

    public static void Copy(T[] sourceArray, int sourceIndex, SparseVector<T> destVector, int destIndex, int length = -1)
    {
      var maxLength = length == -1
        ? sourceArray.Length - sourceIndex
        : Math.Min(length, sourceArray.Length - sourceIndex);
      for (var i = 0; i < maxLength; i++)
      {
        destVector[destIndex + i] = sourceArray[sourceIndex + i];
      }
    }

    public static void Copy(SparseVector<T> sourceVector, SparseVector<T> destVector, int length = -1)
    {
      Copy(sourceVector, 0, destVector, 0, length);
    }

    public static void Copy(SparseVector<T> sourceVector, int sourceIndex, SparseVector<T> destVector, int destIndex, int length = -1)
    {
      var maxLength = length == -1
        ? sourceVector.Length - sourceIndex
        : Math.Min(length, sourceVector.Length - sourceIndex);
      for (var i = 0; i < maxLength; i++)
      {
        destVector[destIndex + i] = sourceVector[sourceIndex + i];
      }
    }
  }
}