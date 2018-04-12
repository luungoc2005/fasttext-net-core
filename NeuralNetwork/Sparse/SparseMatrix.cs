using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Linq;

namespace BotBotNLP.NeuralNetwork.Sparse
{
  public class SparseMatrix<T> // Sparse Matrix in LIL-rows format
  {
    public int Rows {get; private set;}
    public int Cols {get; private set;}
    
    public List<int> RowIndice {get; private set;} = new List<int>();
    public List<List<int>> ColIndice {get; private set;} = new List<List<int>>();
    public List<List<T>> Values {get; private set;} = new List<List<T>>();

    public SparseMatrix(int rows, int cols) {
      this.Rows = rows;
      this.Cols = cols;
    }

    public T this[int row, int col] {
      get {
        return this.Get(row, col);
      }
      set {
        this.Set(row, col, value);
      }
    }

    public SparseVector<T> this[int row] {
      get {
        var result = new SparseVector<T>(Cols);
        if (RowIndice.BinarySearch(row) >= 0) {
          result.Keys.AddRange(ColIndice[row]);
          result.Values.AddRange(Values[row]);
        }
        return result;
      }
      set {
        var rowIndex = RowIndice.BinarySearch(row);
        if (rowIndex < 0) {
          rowIndex = ~rowIndex;
          RowIndice.Insert(rowIndex, row);
          ColIndice.Insert(rowIndex, new List<int>());
          Values.Insert(rowIndex, new List<T>());
        }
        else {
          ColIndice[rowIndex].Clear();
          Values[rowIndex].Clear();
        }
        ColIndice[rowIndex].AddRange(value.Keys);
        Values[rowIndex].AddRange(value.Values);
      }
    }

    private void Set(int row, int col, T value) {
      if (RowIndice.Count == 0) {
        RowIndice.Add(row);
        ColIndice.Add(new List<int>());
        ColIndice[0].Add(col);
        Values.Add(new List<T>());
        Values[0].Add(value);
      }
      else {
        var rowIndex = RowIndice.BinarySearch(row);
        if (rowIndex >= 0) {
          var colIndex = ColIndice[rowIndex].BinarySearch(col);
          if (colIndex >= 0) {
            Values[rowIndex][colIndex] = value;
          }
          else {
            ColIndice[rowIndex].Insert(~colIndex, col);
            Values[rowIndex].Insert(~colIndex, value);
          }
        }
        else {
          var insertIdx = ~rowIndex;
          RowIndice.Insert(insertIdx, row);
          ColIndice.Insert(insertIdx, new List<int>());
          Values.Insert(insertIdx, new List<T>());
          ColIndice[insertIdx].Add(col);
          Values[insertIdx].Add(value);
        }
      }
    }

    private T Get(int row, int col) {
      var rowIndex = RowIndice.BinarySearch(row);
      if (rowIndex >= 0) {
        var colIndex = ColIndice[rowIndex].BinarySearch(col);
        if (colIndex >= 0) {
          return Values[rowIndex][colIndex];
        }
      }
      return default(T);
    }

    // public void CopyFromArray(T[] sourceArray, int sourceIndex, int destIndex, int length) {
    
    // }
  }
}