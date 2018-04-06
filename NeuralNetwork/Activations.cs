using System;
using System.Runtime.CompilerServices;
using System.Linq;

namespace BotBotNLP.NeuralNetwork
{
  public class Activations {
    private static double[] performIter(double[] input, Func<double, double> action) {
      var output = new double[input.Length - 1];
      for (var i = 0; i < input.Length; i++) {
        output[i] = action(input[i]);
      }
      return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ReLU(double input) {
      return Math.Max(input, 0);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] ReLU(double[] input) {
      return performIter(input, ReLU);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ReLU_backwards(double input) {
      return input > 0 ? 1 : 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] ReLU_backwards(double[] input) {
      return performIter(input, ReLU_backwards);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sigmoid(double input) {
      return 1 / (1 + Math.Exp(-input));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Sigmoid(double[] input) {
      return performIter(input, Sigmoid);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Sigmoid_backwards(double input) {
      var s = Sigmoid(input);
      return (1 - s) * s;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Sigmoid_backwards(double[] input) {
      return performIter(input, Sigmoid_backwards);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Tanh(double input) {
      return Math.Tanh(input);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Tanh(double[] input) {
      return performIter(input, Tanh);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Tanh_backwards(double input) {
      var s = Tanh(input);
      return 1 - s * s;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Tanh_backwards(double[] input) {
      return performIter(input, Tanh_backwards);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Softmax(double[] input) {
      // LINQ version
      // var max = input.Max();
      // var scale = input.Select(i => Math.Exp(i - max)).Sum();
      // return input.Select(i => Math.Exp(i - max) / scale).ToArray();
      
      var max = input[0];
      for (var i = 1; i < input.Length; i++) {
        if (input[i] > max) max = input[i];
      }

      var e_x = new double[input.Length];
      var scale = 0.0;
      for (var i = 0; i < input.Length; i++) {
        e_x[i] = Math.Exp(input[i] - max);
        scale += e_x[i];
      }

      for (var i = 0; i < input.Length; i++) {
        e_x[i] = e_x[i] / scale;
      }

      return e_x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] LogSoftmax(double[] input) {
      // LINQ version
      // var max = input.Max();
      // var scale = input.Select(i => Math.Exp(i - max)).Sum();
      // return input.Select(i => Math.Log(Math.Exp(i - max) / scale)).ToArray();

      var max = input[0];
      for (var i = 1; i < input.Length; i++) {
        if (input[i] > max) max = input[i];
      }

      var e_x = new double[input.Length];
      var scale = 0.0;
      for (var i = 0; i < input.Length; i++) {
        e_x[i] = Math.Exp(input[i] - max);
        scale += e_x[i];
      }

      for (var i = 0; i < input.Length; i++) {
        e_x[i] = Math.Log(e_x[i] / scale);
      }

      return e_x;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double[] Softmax_backwards(double[] input) {
      // return input.Select(i => (1 - i) * i).ToArray();
      var e_x = new double[input.Length];
      for (var i = 0; i < input.Length; i++) {
        e_x[i] = (1 - input[i]) * input[i];
      }
      return e_x;
    }
  }
}