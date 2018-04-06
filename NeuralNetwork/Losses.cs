using System;
using System.Runtime.CompilerServices;
using System.Linq;

namespace BotBotNLP.NeuralNetwork
{
  public class Losses {
    public static double NegativeLogLikelihood(double[] input, double[] target) {
      double sum = 0;
      for (var i = 0; i < input.Length; i++) {
        if (target[i] > 0) {
          sum += -Math.Log(input[i]);
        }
      }
      return sum;
    }
  }
}