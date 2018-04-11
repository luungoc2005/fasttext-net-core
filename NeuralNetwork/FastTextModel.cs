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
  }
}