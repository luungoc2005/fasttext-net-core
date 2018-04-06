using System;
using System.Diagnostics;

namespace BotBotNLP.NeuralNetwork
{
  public class FastTextModel {
    public FastTextModel(Int32 inputFeatures, Int32 outputFeatures, Int32 n_hiddens) {

    }

    private string FormatTimespan(TimeSpan ts) {
      return String.Format("{0:00}:{1:00}:{2:00}.{3:00}",
        ts.Hours, ts.Minutes, ts.Seconds,
        ts.Milliseconds / 10);
    }

    public void TrainNetwork(double[][] inputs, double[][] outputs, Int32 epochs = 50, double learningRate = 0.1) {
      var stopwatch = new Stopwatch();

      stopwatch.Start();

      for (var epoch = 1; epoch < epochs + 1; epoch++) {

      }
      stopwatch.Stop();

      Console.WriteLine("Training finished in {0}", 
        FormatTimespan(stopwatch.Elapsed));
    }

    public double[] Predict(double[] input) {
      return new double[input.Length - 1];
    }
  }
}