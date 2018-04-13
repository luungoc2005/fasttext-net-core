using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Linq;
using BotBotNLP.NeuralNetwork.Sparse;

namespace BotBotNLP.NeuralNetwork
{
  public class SparseNetwork
  {
    private static Random rnd;

    private int numInput;
    private int numHidden;
    private int numOutput;

    private SparseVector<double> inputs;

    private SparseMatrix<double> ihWeights; // input-hidden
    private SparseVector<double> hBiases;
    private SparseVector<double> hOutputs;

    private SparseMatrix<double> hoWeights; // hidden-output
    private SparseVector<double> oBiases;

    private SparseVector<double> outputs;

    // back-prop specific arrays (these could be local to method UpdateWeights)
    private SparseVector<double> oGrads; // output gradients for back-propagation
    private SparseVector<double> hGrads; // hidden gradients for back-propagation

    // back-prop momentum specific arrays (these could be local to method Train)
    // private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
    // private double[] hPrevBiasesDelta;
    // private double[][] hoPrevWeightsDelta;
    // private double[] oPrevBiasesDelta;

    // Adam parameters
    public double alpha {get; set;} = 0.001;
    public double beta1 {get; set;} = 0.9;
    public double beta2 {get; set;} = 0.999;
    public double epsilon {get; set;} = 1e-8;
    private SparseMatrix<double> hGrads_m;
    private SparseMatrix<double> hGrads_v;
    private SparseVector<double> hBiases_m;
    private SparseVector<double> hBiases_v;
    private SparseMatrix<double> oGrads_m;
    private SparseMatrix<double> oGrads_v;
    private SparseVector<double> oBiases_m;
    private SparseVector<double> oBiases_v;

    public SparseNetwork(int numInput, int numHidden, int numOutput)
    {
      rnd = new Random(197); // for InitializeWeights() and Shuffle()

      this.numInput = numInput;
      this.numHidden = numHidden;
      this.numOutput = numOutput;

      this.inputs = new SparseVector<double>(numInput);

      this.ihWeights = new SparseMatrix<double>(numInput, numHidden);
      this.hBiases = new SparseVector<double>(numHidden);
      this.hOutputs = new SparseVector<double>(numHidden);

      this.hoWeights = new SparseMatrix<double>(numHidden, numOutput);
      this.oBiases = new SparseVector<double>(numOutput);

      this.outputs = new SparseVector<double>(numOutput);

      // back-prop related arrays below
      this.hGrads = new SparseVector<double>(numHidden);
      this.oGrads = new SparseVector<double>(numOutput);

      // Adam related arrays
      this.hGrads_m = new SparseMatrix<double>(numInput, numHidden);
      this.hGrads_v = new SparseMatrix<double>(numInput, numHidden);
      this.hBiases_m = new SparseVector<double>(numHidden);
      this.hBiases_v = new SparseVector<double>(numHidden);
      this.oGrads_m = new SparseMatrix<double>(numHidden, numOutput);
      this.oGrads_v = new SparseMatrix<double>(numHidden, numOutput);
      this.oBiases_m = new SparseVector<double>(numHidden);
      this.oBiases_v = new SparseVector<double>(numHidden);

      // this.ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
      // this.hPrevBiasesDelta = new double[numHidden];
      // this.hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
      // this.oPrevBiasesDelta = new double[numOutput];
    }

    // public override string ToString()
    // {
    //   string s = "";
    //   s += "===============================\n";
    //   s += "numInput = " + numInput + " numHidden = " + numHidden + " numOutput = " + numOutput + "\n\n";

    //   s += "inputs: \n";
    //   for (int i = 0; i < inputs.Length; ++i)
    //     s += inputs[i].ToString("F2") + "\t";
    //   s += "\n\n";

    //   s += "ihWeights: \n";
    //   for (int i = 0; i < ihWeights.Length; ++i)
    //   {
    //     for (int j = 0; j < ihWeights[i].Length; ++j)
    //     {
    //       s += ihWeights[i][j].ToString("F4") + "\t";
    //     }
    //     s += "\n";
    //   }
    //   s += "\n";

    //   s += "hBiases: \n";
    //   for (int i = 0; i < hBiases.Length; ++i)
    //     s += hBiases[i].ToString("F4") + "\t";
    //   s += "\n\n";

    //   s += "hOutputs: \n";
    //   for (int i = 0; i < hOutputs.Length; ++i)
    //     s += hOutputs[i].ToString("F4") + "\t";
    //   s += "\n\n";

    //   s += "hoWeights: \n";
    //   for (int i = 0; i < hoWeights.Length; ++i)
    //   {
    //     for (int j = 0; j < hoWeights[i].Length; ++j)
    //     {
    //       s += hoWeights[i][j].ToString("F4") + "\t";
    //     }
    //     s += "\n";
    //   }
    //   s += "\n";

    //   s += "oBiases: \n";
    //   for (int i = 0; i < oBiases.Length; ++i)
    //     s += oBiases[i].ToString("F4") + "\t";
    //   s += "\n\n";

    //   s += "outputs: \n";
    //   for (int i = 0; i < outputs.Length; ++i)
    //     s += outputs[i].ToString("F2") + "\t";
    //   s += "\n\n";

    //   s += "===============================\n";
    //   return s;
    // }

    // public void SetWeights(double[] weights)
    // {
    //   // copy weights and biases in weights[] array to i-h weights, i-h biases, h-o weights, h-o biases
    //   int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    //   if (weights.Length != numWeights)
    //     throw new Exception("Bad weights array length: ");

    //   int k = 0; // points into weights param

    //   for (int i = 0; i < numInput; ++i)
    //     for (int j = 0; j < numHidden; ++j)
    //       ihWeights[i][j] = weights[k++];
    //   for (int i = 0; i < numHidden; ++i)
    //     hBiases[i] = weights[k++];
    //   for (int i = 0; i < numHidden; ++i)
    //     for (int j = 0; j < numOutput; ++j)
    //       hoWeights[i][j] = weights[k++];
    //   for (int i = 0; i < numOutput; ++i)
    //     oBiases[i] = weights[k++];
    // }

    public void InitializeWeights()
    {
      // Gaussian initialization
      // int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
      // double[] initialWeights = new double[numWeights];
      // for (int i = 0; i < initialWeights.Length; ++i)
      //   initialWeights[i] = NextGaussian(0, 0.01);
      // this.SetWeights(initialWeights);

      // Xavier initialization
      // http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
      // 2/var because of ReLU

      // var ihStdVar = Math.Sqrt(2d / numInput);
      // for (int i = 0; i < numInput; ++i)
      // {
      //   for (int j = 0; j < numHidden; ++j) 
      //   {
      //     ihWeights[i, j] = NextGaussian(0, ihStdVar);
      //   }
      // }
      // var hBiasesStdVar = Math.Sqrt(2d / numHidden);
      // for (int i = 0; i < numHidden; ++i)
      // {
      //   hBiases[i] = NextGaussian(0, hBiasesStdVar);
      // }
      // var hoStdVar = Math.Sqrt(2d / numHidden);
      // for (int i = 0; i < numHidden; ++i)
      // {
      //   for (int j = 0; j < numOutput; ++j)
      //   {
      //     hoWeights[i, j] = NextGaussian(0, hoStdVar);
      //   }
      // }
      // var oBiasesStdVar = Math.Sqrt(2d / numOutput);
      // for (int i = 0; i < numOutput; ++i)
      // {
      //   oBiases[i] = NextGaussian(0, oBiasesStdVar);
      // }
    }

    private const double TwoPI = 2d * Math.PI;
    private double NextGaussian(double mean, double stdDev) {
      var u1 = 1d - rnd.NextDouble(); //uniform(0,1] random doubles
      var u2 = 1d - rnd.NextDouble();
      var randStdNormal = Math.Sqrt(-2d * Math.Log(u1)) * Math.Sin(TwoPI * u2); //random normal(0,1)
      return mean + stdDev * randStdNormal;
    }

    // public double[] GetWeights()
    // {
    //   // returns the current set of wweights, presumably after training
    //   int numWeights = (numInput * numHidden) + (numHidden * numOutput) + numHidden + numOutput;
    //   double[] result = new double[numWeights];
    //   int k = 0;
    //   for (int i = 0; i < ihWeights.Rows; ++i)
    //     for (int j = 0; j < ihWeights.Cols; ++j)
    //       result[k++] = ihWeights[i, j];
    //   for (int i = 0; i < hBiases.Length; ++i)
    //     result[k++] = hBiases[i];
    //   for (int i = 0; i < hoWeights.Rows; ++i)
    //     for (int j = 0; j < hoWeights.Cols; ++j)
    //       result[k++] = hoWeights[i, j];
    //   for (int i = 0; i < oBiases.Length; ++i)
    //     result[k++] = oBiases[i];
    //   return result;
    // }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public SparseVector<double> ComputeOutputs(SparseVector<double> xValues)
    {
      if (xValues.Length != numInput)
        throw new Exception($"Bad xValues array length. Got {xValues.Length}, expected {numInput}");

      var hSums = new SparseVector<double>(numHidden); // hidden nodes sums scratch array
      var oSums = new SparseVector<double>(numOutput); // output nodes sums

      this.inputs = xValues; // copy x-values to inputs

      for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
      {
        for (int i = 0; i < this.inputs.Keys.Count; i++)
        {
          hSums[j] += this.inputs.Values[i] * this.ihWeights[this.inputs.Keys[i], j]; // note +=
        }
        
        // add biases to input-to-hidden sums
        hSums[j] += this.hBiases[j];
        // this.hOutputs[i] = HyperTanFunction(hSums[i]); // Apply activation
        this.hOutputs[j] = ReLUFunction(hSums[j]); // Apply activation
      }

      for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
      {
        for (int i = 0; i < numHidden; ++i)
          oSums[j] += hOutputs[i] * hoWeights[i, j];
        oSums[j] += oBiases[j];
      }

      var softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
      this.outputs = softOut;

      return softOut;
    } // ComputeOutputs

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double HyperTanFunction(double x)
    {
      if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
      else if (x > 20.0) return 1.0;
      else return Math.Tanh(x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double ReLUFunction(double x) 
    {
      return Math.Max(0, x);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static SparseVector<double> Softmax(SparseVector<double> oSums) 
    {
      // does all output nodes at once so scale doesn't have to be re-computed each time
      // 1. determine max output sum
      var max = oSums.Values[0];
      if (oSums.Values.Count > 1) {
        for (int i = 1; i < oSums.Values.Count; i++)
          if (oSums.Values[i] > max) max = oSums[i];
      }

      // 2. determine scaling factor -- sum of exp(each val - max)

      SparseVector<double> result = new SparseVector<double>(oSums.Length);
      var scale = 0d;
      for (int i = 0; i < oSums.Keys.Count; i++)
      {
        result[oSums.Keys[i]] = Math.Exp(oSums.Values[i] - max);
        scale += result[oSums.Keys[i]];
      }

      for (int i = 0; i < oSums.Values.Count; ++i)
        result[oSums.Keys[i]] = result[oSums.Keys[i]] / scale;

      return result; // now scaled so that xi sum to 1.0
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void UpdateWeights(SparseVector<double> tValues, int epoch, double learnRate, bool initWeights = false)
    {
      var ihStdVar = 0d;
      var hBiasesStdVar = 0d;
      var hoStdVar = 0d;
      var oBiasesStdVar = 0d;
      if (initWeights) {
        ihStdVar = Math.Sqrt(2d / numInput);
        hBiasesStdVar = Math.Sqrt(2d / numHidden);
        hoStdVar = Math.Sqrt(2d / numHidden);
        oBiasesStdVar = Math.Sqrt(2d / numOutput);
      }
      // update the weights and biases using back-propagation, with target values, eta (learning rate),
      // alpha (momentum)
      // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays and
      // matrices have values (other than 0.0)
      if (tValues.Length != numOutput)
        throw new Exception("target values not same Length as output in UpdateWeights");

      // 1. compute output gradients
      for (int i = 0; i < tValues.Length; ++i)
      {
        // MSE version
        // derivative of softmax = (1 - y) * y (same as log-sigmoid)
        // var derivative = (1 - outputs[i]) * outputs[i];
        // oGrads[i] = derivative * (tValues[i] - outputs[i]);
        // cross entropy version
        oGrads[i] = (tValues[i] - outputs[i]);
      }

      // 2. compute hidden gradients
      for (int i = 0; i < hOutputs.Keys.Count; ++i)
      {
        // derivative of tanh = (1 - y) * (1 + y)
        // double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
        // derivative of ReLU = y > 0 ? 1 : 0
        var i1 = hOutputs.Keys[i];
        var derivative = hOutputs[i1] > 0 ? 1 : 0;
        var sum = 0d;
        for (int j = 0; j < oGrads.Length; ++j) // each hidden delta is the sum of numOutput terms
        {
          if (initWeights && hoWeights[i1, j] == 0d) hoWeights[i1, j] = NextGaussian(0, hoStdVar);
          sum += oGrads[j] * hoWeights[i1, j];
        }
        hGrads[i1] = derivative * sum;
      }

      // 3a. update hidden weights (gradients must be computed right-to-left but weights
      // can be updated in any order)
      for (var i = 0; i < inputs.Keys.Count; ++i) // 0..2 (3)
      {
        var i1 = inputs.Keys[i];
        for (var j = 0; j < ihWeights.Cols; ++j) // 0..3 (4)
        {
          if (initWeights && ihWeights[i1, j] == 0d) ihWeights[i1, j] = NextGaussian(0, ihStdVar);
          
          var gradient = hGrads[j] * inputs.Values[i];
          // Adam update rule
          
          hGrads_m[i1, j] = (beta1 * hGrads_m[i1, j]) + (1d - beta1) * gradient;
          hGrads_v[i1, j] = (beta2 * hGrads_v[i1, j]) + (1d - beta2) * gradient * gradient;

          var m_cap = hGrads_m[i1, j] / (1d - Math.Pow(beta1, epoch));
          var v_cap = hGrads_v[i1, j] / (1d - Math.Pow(beta2, epoch));

          var delta = (learnRate * m_cap) / (Math.Sqrt(v_cap) + epsilon);
          // var delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
          ihWeights[i1, j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
          // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
          // ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j]; 
          // weight decay would go here
          // ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
        }
      }

      // 3b. update hidden biases
      for (var i = 0; i < hBiases.Length; ++i)
      {
        hBiases_m[i] = (beta1 * hBiases_m[i]) + (1d - beta1) * hGrads[i];
        hBiases_v[i] = (beta2 * hBiases_v[i]) + (1d - beta2) * hGrads[i] * hGrads[i];

        var m_cap = hBiases_m[i] / (1d - Math.Pow(beta1, epoch));
        var v_cap = hBiases_v[i] / (1d - Math.Pow(beta2, epoch));

        var delta = (learnRate * m_cap) / (Math.Sqrt(v_cap) + epsilon) * hOutputs[i];

        // the 1.0 below is the constant input for any bias; could leave out
        // var delta = learnRate * hGrads[i] * 1.0; 
        hBiases[i] += delta;
        // hBiases[i] += momentum * hPrevBiasesDelta[i]; // momentum
        // weight decay here
        // hPrevBiasesDelta[i] = delta; // don't forget to save the delta
      }

      // 4. update hidden-output weights
      for (var i = 0; i < hoWeights.Rows; ++i)
      {
        for (var j = 0; j < hoWeights.Cols; ++j)
        {
          var gradient = oGrads[j] * hOutputs[i];
          // see above: hOutputs are inputs to the nn outputs
          oGrads_m[i, j] = (beta1 * oGrads_m[i, j]) + (1d - beta1) * gradient;
          oGrads_v[i, j] = (beta2 * oGrads_v[i, j]) + (1d - beta2) * gradient * gradient;

          var m_cap = oGrads_m[i, j] / (1d - Math.Pow(beta1, epoch));
          var v_cap = oGrads_v[i, j] / (1d - Math.Pow(beta2, epoch));

          var delta = (learnRate * m_cap) / (Math.Sqrt(v_cap) + epsilon);

          // var delta = learnRate * oGrads[j] * hOutputs[i];
          hoWeights[i, j] += delta;
          // hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
          // weight decay here
          // hoPrevWeightsDelta[i][j] = delta; // save
        }
      }

      // 4b. update output biases
      for (var i = 0; i < oBiases.Length; ++i)
      {
        oBiases_m[i] = (beta1 * oBiases_m[i]) + (1d - beta1) * oGrads[i];
        oBiases_v[i] = (beta2 * oBiases_v[i]) + (1d - beta2) * oGrads[i] * oGrads[i];

        var m_cap = oBiases_v[i] / (1d - Math.Pow(beta1, epoch));
        var v_cap = oBiases_v[i] / (1d - Math.Pow(beta2, epoch));

        var delta = (learnRate * m_cap) / (Math.Sqrt(v_cap) + epsilon) * hOutputs[i];

        // var delta = learnRate * oGrads[i] * 1.0;
        oBiases[i] += delta;
        // oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
        // weight decay here
        // oPrevBiasesDelta[i] = delta; // save
      }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Train(Tuple<SparseMatrix<double>, SparseMatrix<double>> trainData,
      int maxEprochs,
      double learnRate,
      bool shuffle = true)
    {
      // train a back-prop style NN classifier using learning rate and momentum
      // no weight decay
      var epoch = 1;

      var xValues = trainData.Item1; // inputs
      var tValues = trainData.Item2; // target values

      var sequence = Enumerable.Range(0, xValues.Rows).ToArray();

      var lossHistory = new List<double>();

      while (epoch <= maxEprochs)
      {
        var loss = NLLLoss(trainData);
        lossHistory.Add(loss);
        Console.WriteLine($"Iteration {epoch} - Loss {loss}");

        Shuffle(sequence); // visit each training data in random order
        for (int i = 0; i < xValues.Rows; ++i)
        {
          int idx = sequence[i];
          ComputeOutputs(xValues[idx]); // copy xValues in, compute outputs (and store them internally)
          UpdateWeights(tValues[idx], epoch, learnRate, epoch == 1); // use back-prop to find better weights
        }

        // Check for early stopping
        if (lossHistory.Count > 3) { // For a minimum of 3 epochs
          if (Math.Abs(lossHistory[epoch - 3] - lossHistory[epoch - 1]) < 0.001) // Loss increasing for 2 consecutive times
          {
            Console.WriteLine("Early stopping.");
            break;
          }
        }
        ++epoch;
      }
    }

    private static void Shuffle(int[] sequence)
    {
      for (var i = 0; i < sequence.Length; ++i)
      {
        var r = rnd.Next(i, sequence.Length);
        var tmp = sequence[r];
        sequence[r] = sequence[i];
        sequence[i] = tmp;
      }
    }

    // private double MeanSquaredError(double[][] trainData) // used as a training stopping condition
    // {
    //   // average squared error per training tuple
    //   double sumSquaredError = 0.0;
    //   double[] xValues = new double[numInput]; // first numInput values in trainData
    //   double[] tValues = new double[numOutput]; // last numOutput values

    //   for (int i = 0; i < trainData.Length; ++i) 
    //   {
    //     // walk thru each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
    //     //  where the parens are not really there
    //     Array.Copy(trainData[i], xValues, numInput); // get xValues.
    //     Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
    //     double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
    //     for (int j = 0; j < numOutput; ++j)
    //     {
    //       double err = tValues[j] - yValues[j];
    //       sumSquaredError += err * err;
    //     }
    //   }

    //   return sumSquaredError / trainData.Length;
    // }

    private double NLLLoss(Tuple<SparseMatrix<double>, SparseMatrix<double>> trainData) // used as a training stopping condition
    {
      // average squared error per training tuple
      var nllloss = 0.0;
      var xValues = trainData.Item1; // first numInput values in trainData

      for (var i = 0; i < xValues.Rows; ++i) 
      {
        var yValues = this.ComputeOutputs(xValues[i]); // compute output using current weights
        var tValues = trainData.Item2[i];

        for (var j = 0; j < numOutput; ++j)
        {
          var err = -(tValues[j] * Math.Log(yValues[j] + 1e-8));
          nllloss += err;
        }
      }

      return nllloss / xValues.Rows;
    }

    public double Accuracy(Tuple<SparseMatrix<double>, SparseMatrix<double>> testData)
    {
      // percentage correct using winner-takes all
      var numCorrect = 0;
      var numWrong = 0;
      var xValues = testData.Item1; // inputs
      var tValues = testData.Item2; // targets
      SparseVector<double> yValues; // computed Y

      for (var i = 0; i < xValues.Rows; ++i)
      {
        yValues = this.ComputeOutputs(xValues[i]);
        var maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

        if (tValues[i][maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
          ++numCorrect;
        else
          ++numWrong;
      }
      return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
    }

    public static int MaxIndex(SparseVector<double> vector) // helper for Accuracy()
    {
      // index of largest value
      var bigIndex = 0;
      var biggestVal = vector.Values[0];
      for (var i = 1; i < vector.Values.Count; ++i)
      {
        if (vector.Values[i] > biggestVal)
        {
          biggestVal = vector.Values[i]; bigIndex = vector.Keys[i];
        }
      }
      return bigIndex;
    }
  }
}