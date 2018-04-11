using System;
using System.IO;
using System.Diagnostics;
using System.Linq;
using BotBotNLP.Vectorizers;
using BotBotNLP.NeuralNetwork;
using BotBotNLP.NeuralNetwork.Tests;

namespace BotBotNLP
{
    class Program
    {
        static void Main(string[] args)
        {
            var glovePath = Path.Combine(Directory.GetCurrentDirectory(), "data/glove.6B.300d.txt");
            var stopwatch = new Stopwatch();

            double[][] trainData = null;
            double[][] testData = null;
            MakeTrainTest(Iris.Dataset(), out trainData, out testData);

            stopwatch.Start();
            
            const int numInput = 4;
            const int numHidden = 7;
            const int numOutput = 3;
            var nn = new LinearNetwork(numInput, numHidden, numOutput);
            nn.InitializeWeights();
            // Console.WriteLine(nn.ToString());

            var maxEpochs = 50;
            double learnRate = 0.02;

            nn.Train(trainData, maxEpochs, learnRate);

            stopwatch.Stop();
            Console.WriteLine("Task finished in {0} ms", stopwatch.ElapsedMilliseconds);

            var trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            var testAcc = nn.Accuracy(testData);
            Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
            
            Console.WriteLine(nn.ToString());
            // Console.WriteLine("Loading GLoVE vectors from {0}", glovePath);
            // stopwatch.Start();
            // var reader = new WordVectorReader(glovePath);
            // stopwatch.Stop();

            // var vectorizer = new SentenceVectorizer(reader);

            // while (true) {
            //     Console.Write("Word for inference: ");
            //     var sentence = Console.ReadLine();

            //     if (sentence == "exit") break;

            //     stopwatch.Restart();
            //     var sent_vector = vectorizer.SentenceToVector(sentence, false);
            //     stopwatch.Stop();

            //     Console.WriteLine("Result: {0}", String.Join(",", sent_vector.Take(100)));
            //     Console.WriteLine("Inference time: {0}", stopwatch.ElapsedMilliseconds);
            // }
        }


        static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData)
        {
            // split allData into 80% trainData and 20% testData
            Random rnd = new Random(0);
            int totRows = allData.Length;
            int numCols = allData[0].Length;

            int trainRows = (int)(totRows * 0.80); // hard-coded 80-20 split
            int testRows = totRows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            int[] sequence = new int[totRows]; // create a random sequence of indexes
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // index into sequence[]
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) // first rows to train data
            {
                trainData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], numCols);
                ++j;
            }

            j = 0; // reset to start of test data
            for (; si < totRows; ++si) // remainder to test data
            {
                testData[j] = new double[numCols];
                int idx = sequence[si];
                Array.Copy(allData[idx], testData[j], numCols);
                ++j;
            }
        }
    }
}
