using System;
using System.IO;
using System.Diagnostics;
using System.Linq;
using BotBotNLP.Vectorizers;
using BotBotNLP.NeuralNetwork;

namespace BotBotNLP
{
    class Program
    {
        static void Main(string[] args)
        {
            var glovePath = Path.Combine(Directory.GetCurrentDirectory(), "data/glove.6B.300d.txt");
            var stopwatch = new Stopwatch();

            Console.WriteLine("Loading GLoVE vectors from {0}", glovePath);
            stopwatch.Start();
            var reader = new WordVectorReader(glovePath);
            stopwatch.Stop();
            Console.WriteLine("Task finished in {0} ms", stopwatch.ElapsedMilliseconds);

            var vectorizer = new SentenceVectorizer(reader);

            while (true) {
                Console.Write("Word for inference: ");
                var sentence = Console.ReadLine();

                if (sentence == "exit") break;

                stopwatch.Restart();
                var sent_vector = vectorizer.SentenceToVector(sentence, false);
                stopwatch.Stop();

                Console.WriteLine("Result: {0}", String.Join(",", sent_vector.Take(100)));
                Console.WriteLine("Inference time: {0}", stopwatch.ElapsedMilliseconds);
            }
        }
    }
}
