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
            var dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data/kc_data.json");
            
            var stopwatch = new Stopwatch();

            // double[][] trainData = null;
            // double[][] testData = null;
            // MakeTrainTest(Iris.Dataset(), out trainData, out testData);

            Console.WriteLine("Loading GLoVE vectors from {0}", glovePath);
            stopwatch.Start();
            var reader = new WordVectorReader(glovePath);
            stopwatch.Stop();
            Console.WriteLine("Task finished in {0} ms", stopwatch.ElapsedMilliseconds);

            stopwatch.Restart();
            var vectorizer = new SentenceVectorizer(reader);
            var loader = new DataLoader.IntentsLoader(dataPath, vectorizer);

            var trainData = loader.GetData();
            stopwatch.Stop();
            Console.WriteLine("Task finished in {0} ms", stopwatch.ElapsedMilliseconds);

            Console.WriteLine("Training started");
            stopwatch.Restart();
            
            var numInput = loader.InputDimensions;
            var numHidden = 50;
            var numOutput = loader.OutputDimensions;

            var nn = new SparseNetwork(numInput, numHidden, numOutput);

            var maxEpochs = 50;
            double learnRate = 0.02;

            nn.Train(trainData, maxEpochs, learnRate);
            var trainAcc = nn.Accuracy(trainData);
            Console.WriteLine("\nAccuracy on training data = " + trainAcc.ToString("F4"));

            stopwatch.Stop();
            Console.WriteLine("Task finished in {0} ms", stopwatch.ElapsedMilliseconds);

            // var testAcc = nn.Accuracy(testData);
            // Console.WriteLine("\nAccuracy on test data = " + testAcc.ToString("F4"));
            
            // Console.WriteLine(nn.ToString());

            // var test = new NeuralNetwork.Sparse.SparseMatrix<double>(10000000, 10000000);
            // test[2, 4] = 1d;
            // test[10000, 20] = 2d;
            // test[10000, 30] = 4d;
            // test[5000000, 5000000] = Math.PI;
            // Console.WriteLine(test[2, 4]);
            // Console.WriteLine(test[10000, 20]);
            // Console.WriteLine(test[10000, 30]);
            // Console.WriteLine(test[5000000, 5000000]);
            // Console.WriteLine(test[10, 20]);
            
            while (true) {
                Console.Write("Sentence for testing: ");
                var sentence = Console.ReadLine();

                if (sentence == "exit") break;

                stopwatch.Restart();
                var sent_vector = vectorizer.SentenceToVector(sentence);
                var result = nn.ComputeOutputs(sent_vector);
                var resultIdx = SparseNetwork.MaxIndex(result);
                stopwatch.Stop();

                Console.WriteLine($"Result: {loader.Intents[resultIdx].name}, Confidence: {result[resultIdx]}");
                Console.WriteLine($"Inference time: {stopwatch.ElapsedMilliseconds}");
            }
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
