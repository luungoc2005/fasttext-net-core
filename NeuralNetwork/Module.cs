using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace BotBotNLP.NeuralNetwork
{
  public class Network {
    public List<Layer> HiddenLayers {get; private set;} = new List<Layer>();

    public void AddLayer(Layer layer) {
      this.HiddenLayers.Add(layer);
    }

    public void Forward(double[][] inputs) {
      
    }

    public void Backwards(double[][] targets) {

    }
  }
}